import tvm
from diffusers import StableDiffusionPipeline
import web_stable_diffusion.trace as trace
from tvm.relax.frontend import detach_params


def serialize(mod, params, prefix):
    f = mod[prefix]
    params_list = params[prefix]
    param_names = [p.name_hint for p in f.params]

    params_dict = dict(zip(param_names[-len(params_list):], params_list))

    with open("{}.json".format(prefix), "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    tvm.runtime.save_param_dict_to_file(params_dict, "{}.params".format(prefix))


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

clip, params_clip = detach_params(trace.clip_to_text_embeddings(pipe))
unet, params_unet = detach_params(trace.unet_latents_to_noise_pred(pipe, "cuda"))
vae, params_vae = detach_params(trace.vae_to_image(pipe))

serialize(clip, params_clip, "clip")
serialize(unet, params_unet, "unet")
serialize(vae, params_vae, "vae")
