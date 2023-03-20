import tvm
from tvm import relax

from diffusers import StableDiffusionPipeline
import web_stable_diffusion.trace as trace
import web_stable_diffusion.utils as utils

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
torch_dev_key = utils.detect_available_torch_device()
unet = trace.unet_latents_to_noise_pred(pipe, torch_dev_key)

unet = relax.transform.FoldConstant()(unet)
mod, params = relax.frontend.detach_params(unet)

print(mod.script())

# prefix = "unet"

# with open("{}.json".format(prefix), "w") as fo:
#     fo.write(tvm.ir.save_json(mod))

# tvm.runtime.save_param_dict_to_file(params, "{}.params".format(prefix))
