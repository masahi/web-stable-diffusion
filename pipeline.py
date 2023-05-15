import re
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

import numpy as np

import time
import inspect
from typing import List, Optional, Union

import tvm
from tvm import relax

from diffusers import (
    LMSDiscreteScheduler,
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def convert_to_ndarray(tensor):
    return tvm.runtime.ndarray.from_dlpack(to_dlpack(tensor))


def transform_params(vm, params, dev):
    if params:
        params_gpu = [p.copyto(dev) for p in params]
        return vm["main_transform_params"](params_gpu)

    return None


class StableDiffusionTVMPipeline:
    def __init__(
        self,
        original_pipe,
        text_encoder,
        unet,
        vae,
        clip_params=None,
        unet_params=None,
        vae_params=None,
    ):
        self.dev = tvm.device("cuda", 0)

        self.original_pipe = original_pipe
        self.clip = relax.VirtualMachine(text_encoder, self.dev)
        self.vae = relax.VirtualMachine(vae, self.dev)
        self.unet = relax.VirtualMachine(unet, self.dev)
        self.tokenizer = self.original_pipe.tokenizer
        self.scheduler = self.original_pipe.scheduler
        self.safety_checker = self.original_pipe.safety_checker

        self.clip_params = transform_params(self.clip, clip_params, self.dev)
        self.unet_params = transform_params(self.unet, unet_params, self.dev)
        self.vae_params = transform_params(self.vae, vae_params, self.dev)

        # Warm up, for some reason from_dlpack can take > 0.7 sec on first call depending on environment
        inputs = [tvm.nd.array(np.zeros((1, 77)).astype("int64"), self.dev)]
        if self.clip_params:
            inputs.append(self.clip_params)

        from_dlpack(self.clip["main"](*inputs))

    def unet_inference(self, latent_model_input, timesteps, encoder_hidden_states):
        inputs = [
            convert_to_ndarray(latent_model_input),
            tvm.nd.array(timesteps.numpy().astype("int32"), self.dev),
            convert_to_ndarray(encoder_hidden_states),
        ]

        if self.unet_params:
            inputs.append(self.unet_params)

        return from_dlpack(self.unet["main"](*inputs))

    def clip_inference(self, input_ids):
        inputs = [convert_to_ndarray(input_ids)]

        if self.clip_params:
            inputs.append(self.clip_params)

        return from_dlpack(self.clip["main"](*inputs))

    def vae_inference(self, vae_input):
        inputs = [convert_to_ndarray(vae_input)]

        if self.vae_params:
            inputs.append(self.vae_params)

        return from_dlpack(self.vae["main"](*inputs)) / 255

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        batch_size = 1
        assert height == 512 and width == 512

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.clip_inference(text_input.input_ids.to("cuda"))

        do_classifier_free_guidance = guidance_scale > 1.0
        assert do_classifier_free_guidance, "Not implemeted"

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input.input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.clip_inference(uncond_input.input_ids.to("cuda"))
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
            device="cuda",
        )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
            # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        assert not isinstance(self.scheduler, LMSDiscreteScheduler), "Not implemented"

        for _, t in enumerate(
            self.original_pipe.progress_bar(self.scheduler.timesteps)
        ):
            latents_input = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.unet_inference(
                latents_input, t, encoder_hidden_states=text_embeddings
            )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        image = self.vae_inference(latents)
        image = self.original_pipe.numpy_to_pil(image.cpu().numpy())

        has_nsfw_concept = None

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


def test(model):
    hidden_dim = 768
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    dev = tvm.device(target.kind.name, 0)

    inp_0 = tvm.nd.array(np.random.randn(1, 4, 64, 64).astype("float32"), dev)
    inp_1 = tvm.nd.array(np.array(1, "int32"), dev)
    inp_2 = tvm.nd.array(np.random.randn(2, 77, hidden_dim).astype("float32"), dev)

    if model == "unet":
        inputs = [inp_0, inp_1, inp_2]
    elif model == "vae":
        inputs = [inp_0]
    else:
        inputs = [
            tvm.nd.array(
                np.random.randint(low=0, high=1000, size=(1, 77)).astype("int64"), dev
            )
        ]

    # so_name = "{}.so".format(model)
    # ex = tvm.runtime.load_module(so_name)
    # vm = relax.VirtualMachine(ex, dev, profile=True)

    ex, params = load_model_and_params(model)
    vm = relax.VirtualMachine(ex, dev, profile=True)
    transformed_params = transform_params(vm, params, dev)
    inputs.append(transformed_params)

    print(vm.profile("main", *inputs))

    vm.set_input("main", *inputs)
    print(vm.time_evaluator("invoke_stateful", dev, repeat=50)("main"))


def load_model_and_params(prefix):
    mod = tvm.runtime.load_module(f"{prefix}_no_params.so")
    param_dict = tvm.runtime.load_param_dict_from_file(f"{prefix}_fp16.params")

    names = param_dict.keys()
    sorted_names = sorted(names, key=lambda name: int(name[name.rfind("_") + 1 :]))

    return mod, [param_dict[name] for name in sorted_names]


bind_params = False

# test("unet")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs("pcuenq/pokemon-lora")

lora_weights = {}

for k, v in pipe.unet.state_dict().items():
    if "lora" in k:
        lora_weights[k] = v

# for k, v in lora_weights.items():
#     print(k, v.shape)

# pipe = StableDiffusionPipeline.from_pretrained("XpucT/Deliberate")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# torch.manual_seed(1791574510)

# # model_id = "stabilityai/stable-diffusion-2-1-base"
# # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# # pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)

pipe.safety_checker = None
# pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention()


if bind_params:
    clip = tvm.runtime.load_module("clip.so")
    unet = tvm.runtime.load_module("unet.so")
    vae = tvm.runtime.load_module("vae.so")

    pipe_tvm = StableDiffusionTVMPipeline(pipe, clip, unet, vae)
else:
    clip, clip_params = load_model_and_params("clip")
    vae, vae_params = load_model_and_params("vae")
    # unet, unet_params = load_model_and_params("unet")

    param_dict = tvm.runtime.load_param_dict_from_file("unet_fp16.params")

    names = param_dict.keys()
    sorted_names = sorted(names, key=lambda name: int(name[name.rfind("_") + 1 :]))
    unet = tvm.runtime.load_module("unet_no_params.so")

    attention_weights = {}

    for k, v in param_dict.items():
        if "transformer_blocks" in k and not "norm" in k:
            attention_weights[k] = v

    unet_params = []

    for name in sorted_names:
        if "transformer_blocks" in name and "to_" in name and not "bias" in name:
            if "mid" in name:
                match = re.findall(
                    "unet.mid_block.attentions.(\d).transformer_blocks.0.attn(\d).to_([a-z]+)",
                    name,
                )
                assert match
                attn_id, attn_inner_id, matmul_kind = match[0]
                lora_param_name_down = f"mid_block.attentions.{attn_id}.transformer_blocks.0.attn{attn_inner_id}.processor.to_{matmul_kind}_lora.down.weight"
                lora_param_name_up = f"mid_block.attentions.{attn_id}.transformer_blocks.0.attn{attn_inner_id}.processor.to_{matmul_kind}_lora.up.weight"

                assert lora_param_name_down in lora_weights
                assert lora_param_name_up in lora_weights
            else:
                match = re.findall(
                    "unet.([a-z]+)_blocks.(\d).attentions.(\d).transformer_blocks.0.attn(\d).to_([a-z]+)",
                    name,
                )
                assert match
                block_kind, block_id, attn_id, attn_inner_id, matmul_kind = match[0]

                # print(block_kind, block_id, attn_id, attn_inner_id, matmul_kind)
                lora_param_name_down = f"{block_kind}_blocks.{block_id}.attentions.{attn_id}.transformer_blocks.0.attn{attn_inner_id}.processor.to_{matmul_kind}_lora.down.weight"
                lora_param_name_up = f"{block_kind}_blocks.{block_id}.attentions.{attn_id}.transformer_blocks.0.attn{attn_inner_id}.processor.to_{matmul_kind}_lora.up.weight"

                assert lora_param_name_down in lora_weights
                assert lora_param_name_up in lora_weights

            lora_down = lora_weights[lora_param_name_down].numpy()
            lora_up = lora_weights[lora_param_name_up].numpy()
            orig_param = param_dict[name]
            unet_params.append(
                tvm.nd.array(
                    orig_param.numpy() + np.dot(lora_up, lora_down).astype("float16")
                )
            )
        else:
            unet_params.append(param_dict[name])

    pipe_tvm = StableDiffusionTVMPipeline(
        pipe, clip, unet, vae, clip_params, unet_params, vae_params
    )

t1 = time.time()
sample = pipe_tvm("Green pokemon with menacing face", num_inference_steps=25)["images"][
    0
]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
