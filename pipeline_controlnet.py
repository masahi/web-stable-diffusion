import PIL.Image
import cv2

import requests
from io import BytesIO

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
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import load_image


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
        controlnet,
        clip_params=None,
        unet_params=None,
        vae_params=None,
        controlnet_params=None,
    ):
        self.dev = tvm.device("cuda", 0)

        self.original_pipe = original_pipe
        self.clip = relax.VirtualMachine(text_encoder, self.dev)
        self.vae = relax.VirtualMachine(vae, self.dev)
        self.unet = relax.VirtualMachine(unet, self.dev)
        self.controlnet = relax.VirtualMachine(controlnet, self.dev)

        self.tokenizer = self.original_pipe.tokenizer
        self.scheduler = self.original_pipe.scheduler
        self.safety_checker = self.original_pipe.safety_checker

        self.clip_params = transform_params(self.clip, clip_params, self.dev)
        self.unet_params = transform_params(self.unet, unet_params, self.dev)
        self.vae_params = transform_params(self.vae, vae_params, self.dev)
        self.controlnet_params = transform_params(
            self.controlnet, controlnet_params, self.dev
        )

        # Warm up, for some reason from_dlpack can take > 0.7 sec on first call depending on environment
        inputs = [tvm.nd.array(np.zeros((1, 77)).astype("int64"), self.dev)]
        if self.clip_params:
            inputs.append(self.clip_params)

        from_dlpack(self.clip["main"](*inputs))

    def unet_inference(
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        down_block_res_samples,
        mid_block_res_sample,
    ):
        inputs = [
            convert_to_ndarray(latent_model_input),
            tvm.nd.array(timesteps.numpy().astype("int32"), self.dev),
            convert_to_ndarray(encoder_hidden_states),
            (convert_to_ndarray(sample) for sample in down_block_res_samples),
            convert_to_ndarray(mid_block_res_sample),
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

    def controlnet_inference(
        self,
        latent_model_input,
        timesteps,
        text_embeddings,
        image,
        controlnet_conditioning_scale,
    ):
        # TODO
        assert controlnet_conditioning_scale == 1
        inputs = [
            convert_to_ndarray(latent_model_input),
            tvm.nd.array(timesteps.numpy().astype("int32"), self.dev),
            convert_to_ndarray(text_embeddings),
            convert_to_ndarray(image),
        ]

        if self.controlnet_params:
            inputs.append(self.controlnet_params)

        return from_dlpack(self.controlnet["main"](*inputs))

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
        ] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
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

        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size,
            num_images_per_prompt=1,
            device="cuda",
            dtype="float16",
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

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
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            down_block_res_samples, mid_block_res_sample = self.controlnet_inference(
                latent_model_input,
                t,
                text_embeddings,
                image,
                controlnet_conditioning_scale,
            )

            noise_pred = self.unet_inference(
                latent_model_input,
                t,
                text_embeddings,
                down_block_res_samples,
                mid_block_res_sample,
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


def load_model_and_params(prefix):
    mod = tvm.runtime.load_module(f"{prefix}_no_params.so")
    param_dict = tvm.runtime.load_param_dict_from_file(f"{prefix}_fp16.params")

    names = param_dict.keys()
    sorted_names = sorted(names, key=lambda name: int(name[name.rfind("_") + 1 :]))

    return mod, [param_dict[name] for name in sorted_names]


def get_init_image():
    image = load_image(
        "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
    )
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


path = "runwayml/stable-diffusion-v1-5"

bind_params = False

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(path, controlnet=controlnet)

pipe.safety_checker = None
# pipe.to("cuda")

if bind_params:
    clip = tvm.runtime.load_module("clip.so")
    unet = tvm.runtime.load_module("unet.so")
    vae = tvm.runtime.load_module("vae.so")
    controlnet = tvm.runtime.load_module("controlnet.so")

    pipe_tvm = StableDiffusionTVMPipeline(pipe, clip, unet, vae, controlnet)
else:
    clip, clip_params = load_model_and_params("clip")
    vae, vae_params = load_model_and_params("vae")
    unet, unet_params = load_model_and_params("unet")
    controlnet, controlnet_params = load_model_and_params("controlnet")

    pipe_tvm = StableDiffusionTVMPipeline(
        pipe,
        clip,
        unet,
        vae,
        controlnet,
        clip_params,
        unet_params,
        vae_params,
        controlnet_params,
    )

prompt = "bird"
init_image = get_init_image()

t1 = time.time()
sample = pipe(prompt=prompt, image=init_image, num_inference_steps=25)["images"][0]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
