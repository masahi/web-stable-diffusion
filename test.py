import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite_call, is_op, wildcard, is_const
from tvm.script import relax as R
from tvm.relax.transform.tuning_api import Trace


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


def tune_mod(mod):
    work_dir = "work"
    with tvm.transform.PassContext(trace=Trace(mod), opt_level=3):
        mod = relax.transform.MetaScheduleTuneIRMod(
            params={},
            work_dir=work_dir,
            max_trials_global=20000,
        )(mod)
        mod =  relax.transform.MetaScheduleApplyDatabase(work_dir)(mod)
        return tvm.tir.transform.DefaultGPUSchedule()(mod)


with open("unet.json", "r") as fi:
    mod_tmp = tvm.ir.load_json(fi.read())

mod = tvm.IRModule()
mod["main"] = mod_tmp["unet"]
mod["main"] = rewrite_attention(mod["main"])
mod["main"] = simplify_div(mod["main"])

mod = tvm.transform.Sequential([
    relax.transform.EliminateCommonSubexpr(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.CombineParallelMatmul(),
    relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
    # ToMixedPrecision(out_dtype="float16"),
    # relax.transform.LegalizeOps(),
    # relax.transform.FoldConstant(),
    # relax.transform.AnnotateTIROpPattern(),
    # relax.transform.FuseOps(),
    # relax.transform.FuseTIR(),
])(mod)

mod = partition_for_cutlass(mod)
print(mod)
