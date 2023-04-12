from diffusers import StableDiffusionPipeline

import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite_call, is_op, wildcard, is_const
from tvm.script import relax as R


def deserialize(prefix):
    with open("{}.json".format(prefix), "r") as fi:
        mod_orig = tvm.ir.load_json(fi.read())

    mod = tvm.IRModule()
    mod["main"] = mod_orig[prefix]

    params = tvm.runtime.load_param_dict_from_file("{}.params".format(prefix))

    return mod, params


def rewrite_attention(f):
    def BSNH_to_BSH(tensor):
        return is_op("relax.reshape")(is_op("relax.permute_dims")(tensor), wildcard())

    def BSH_to_BSNH(tensor):
        return is_op("relax.permute_dims")(is_op("relax.reshape")(tensor, wildcard()))

    Q = wildcard()
    K = wildcard()
    V = wildcard()

    Q_3D = BSNH_to_BSH(Q)
    V_3D = BSNH_to_BSH(V)
    K_3D = BSNH_to_BSH(K)

    matmul1 = is_op("relax.matmul")(Q_3D, is_op("relax.permute_dims")(V_3D))
    multiply = is_op("relax.multiply")(matmul1, is_const())
    softmax = is_op("relax.nn.softmax")(multiply)
    matmul2 = is_op("relax.matmul")(softmax, K_3D)

    pattern = BSH_to_BSNH(matmul2)

    def callback(_, matchings):
        return R.nn.attention(matchings[Q], matchings[K], matchings[V])

    return rewrite_call(pattern, callback, f)


def simplify_div(f):
    lhs_pat = wildcard()
    rhs_pat = is_const()
    pattern = is_op("relax.divide")(lhs_pat, rhs_pat)

    def is_one(v):
        if isinstance(v, relax.expr.Constant) and v.data.numpy() == 1:
            return True
        return False

    def callback(orig, matchings):
        if is_one(matchings[rhs_pat]):
            return matchings[lhs_pat]
        return orig

    return rewrite_call(pattern, callback, f)


def run_opt_passes(mod, params):
    mod["main"] = rewrite_attention(mod["main"])
    mod["main"] = simplify_div(mod["main"])

    return tvm.transform.Sequential([
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.CombineParallelMatmul(),
        relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
        # ToMixedPrecision(out_dtype="float16"),
        relax.transform.BindParams("main", params),
        # relax.pipeline.get_pipeline(),
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])(mod)


mod_clip, params_clip = deserialize("clip")
mod_unet, params_unet = deserialize("unet")
mod_dec, params_dec = deserialize("vae")

# mod_clip = run_opt_passes(mod_clip, params_clip)
# mod_unet = run_opt_passes(mod_unet, params_unet)
mod_dec = run_opt_passes(mod_dec, params_dec)

print(mod_dec)
