from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

import tvm
import web_stable_diffusion.trace as trace
from tvm.relax.frontend import detach_params


def export_params(pipe, output_prefix):
    clip, params_clip = detach_params(trace.clip_to_text_embeddings(pipe))
    vae, params_vae = detach_params(trace.vae_to_image(pipe))

    if isinstance(pipe, StableDiffusionControlNetPipeline):
        unet, params_unet = detach_params(
            trace.unet_latents_to_noise_pred_controlnet(pipe, "mps")
        )
    else:
        unet, params_unet = detach_params(trace.unet_latents_to_noise_pred(pipe, "mps"))

    def export(mod, params, prefix, output_file):
        params_list = params[prefix]
        param_names = [p.name_hint for p in mod[prefix].params[-len(params_list) :]]
        params_dict = dict(zip(param_names, params_list))
        params_fp16 = {}

        for i, name in enumerate(param_names):
            params_fp16[name + f"_{i}"] = tvm.nd.array(
                params_dict[name].numpy().astype("float16")
            )

        tvm.runtime.save_param_dict_to_file(params_fp16, output_file)

    export(clip, params_clip, "clip", f"{output_prefix}_clip_fp16.params")
    export(vae, params_vae, "vae", f"{output_prefix}_vae_fp16.params")
    export(unet, params_unet, "unet", f"{output_prefix}_unet_fp16.params")


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

export_params(pipe, "sd-controlnet-openpose")
