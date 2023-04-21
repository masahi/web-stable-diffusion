import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

import numpy as np

import time
import inspect
from typing import List, Optional, Union

import tvm
from tvm import relax

from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def convert_to_ndarray(tensor):
    return tvm.runtime.ndarray.from_dlpack(to_dlpack(tensor))


class StableDiffusionTVMPipeline:
    def __init__(
        self,
        original_pipe,
        text_encoder,
        unet,
        vae,
    ):
        self.dev = tvm.device("cuda", 0)

        self.original_pipe = original_pipe
        self.clip = relax.VirtualMachine(text_encoder, self.dev)
        self.vae = relax.VirtualMachine(vae, self.dev)
        self.unet = relax.VirtualMachine(unet, self.dev)
        self.tokenizer = self.original_pipe.tokenizer
        self.scheduler = self.original_pipe.scheduler
        self.safety_checker = self.original_pipe.safety_checker

    def unet_inference(self, latent_model_input, timesteps, encoder_hidden_states):
        out = self.unet["main"](
            # TODO: why it doesn't work
            # convert_to_ndarray(latent_model_input),
            # convert_to_ndarray(timesteps),
            # convert_to_ndarray(encoder_hidden_states),
            tvm.nd.array(latent_model_input.cpu().numpy(), self.dev),
            tvm.nd.array(timesteps.cpu().numpy().astype("int32"), self.dev),
            tvm.nd.array(encoder_hidden_states.cpu().numpy(), self.dev),
        )
        return from_dlpack(out)

    def clip_inference(self, input_ids):
        # TODO: why it doesn't work
        # inp = convert_to_ndarray(input_ids)
        inp = tvm.nd.array(input_ids.numpy(), self.dev)
        tvm_out = self.clip["main"](inp)
        return from_dlpack(tvm_out)

    def vae_inference(self, vae_input):
        # TODO: why it doesn't work
        # inp = convert_to_ndarray(vae_input)
        inp = tvm.nd.array(vae_input.cpu().numpy(), self.dev)
        return from_dlpack(self.vae["main"](inp))

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

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.clip_inference(text_input.input_ids)

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
            uncond_embeddings = self.clip_inference(uncond_input.input_ids)
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

        assert not isinstance(self.scheduler, LMSDiscreteScheduler)

        for i, t in enumerate(
            self.original_pipe.progress_bar(self.scheduler.timesteps)
        ):
            # predict the noise residual
            noise_pred = self.unet_inference(
                latents, t, encoder_hidden_states=text_embeddings
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


def get_result(ex, dev, inputs, time_eval=False):
    vm = relax.VirtualMachine(ex, dev, profile=True)
    out = vm["main"](*inputs)

    if time_eval:
        # print(vm.profile("main", *inputs))

        vm.set_input("main", *inputs)
        print(vm.time_evaluator("invoke_stateful", dev, repeat=50)("main"))

    return out.numpy()


def test(model):
    so_name = "{}.so".format(model)
    ex = tvm.runtime.load_module(so_name)

    target = tvm.target.Target("nvidia/geforce-rtx-3070")

    dev = tvm.device(target.kind.name, 0)
    inp_0 = tvm.nd.array(np.random.randn(1, 4, 64, 64).astype("float32"), dev)
    inp_1 = tvm.nd.array(np.array(1, "int32"), dev)
    inp_2 = tvm.nd.array(np.random.randn(2, 77, 768).astype("float32"), dev)

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

    out = get_result(ex, dev, inputs, time_eval=True)

# test("clip")

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

clip = tvm.runtime.load_module("clip.so")
unet = tvm.runtime.load_module("unet.so")
vae = tvm.runtime.load_module("vae.so")

pipe_tvm = StableDiffusionTVMPipeline(pipe, clip, unet, vae)

t1 = time.time()
sample = pipe_tvm("Mt. Fuji in the style of Gauguin", num_inference_steps=50)["images"][
    0
]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
