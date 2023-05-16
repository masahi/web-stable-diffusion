from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

import tvm
import web_stable_diffusion.trace as trace
from tvm.relax.frontend import detach_params


def serialize(mod, params, prefix):
    f = mod[prefix]
    params_list = params[prefix]
    param_names = [p.name_hint for p in f.params]

    params_dict = dict(zip(param_names[-len(params_list) :], params_list))

    with open("{}.json".format(prefix), "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    tvm.runtime.save_param_dict_to_file(params_dict, "{}.params".format(prefix))


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)

clip, params_clip = detach_params(trace.clip_to_text_embeddings(pipe))
vae, params_vae = detach_params(trace.vae_to_image(pipe))
unet, params_unet = detach_params(trace.unet_latents_to_noise_pred_controlnet(pipe, "mps"))
controlnet, params_controlnet = detach_params(trace.convert_controlnet(pipe))

serialize(clip, params_clip, "clip")
serialize(vae, params_vae, "vae")
serialize(unet, params_unet, "unet")
serialize(controlnet, params_controlnet, "controlnet")
