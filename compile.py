import numpy as np

import tvm
from tvm import relax, tir
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


def run_opt_passes(mod):
    return tvm.transform.Sequential(
        [
            relax.transform.EliminateCommonSubexpr(),
            relax.transform.CanonicalizeBindings(),
            relax.transform.CombineParallelMatmul(),
            relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
            relax.transform.ToMixedPrecision(out_dtype="float16"),
        ]
    )(mod)


def get_lower_passes(params):
    return tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", params),
            relax.pipeline.get_pipeline(),
            tir.transform.DefaultGPUSchedule(),
        ]
    )

def get_result(mod, target, dev, inputs, time_eval=False):
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, dev)
    out = vm["main"](*inputs)

    if time_eval:
        vm.set_input("main", *inputs)
        print(vm.time_evaluator("invoke_stateful", dev, repeat=50)("main"))

    return out.numpy()


def get_ref(mod, params, target, dev, inputs):
    mod = tvm.transform.Sequential(
        [
            relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
            relax.transform.ToMixedPrecision(out_dtype="float16"),
        ]
    )(mod)

    with target:
        mod = get_lower_passes(params)(mod)

    return get_result(mod, target, dev, inputs)


dev = tvm.device("cuda", 0)
inp_0 = tvm.nd.array(np.random.randn(1, 4, 64, 64).astype("float32"), dev)
inp_1 = tvm.nd.array(np.array(1, "int32"), dev)
inp_2 = tvm.nd.array(np.random.randn(2, 77, 768).astype("float32"), dev)
inputs = [inp_0, inp_1, inp_2]
target = tvm.target.Target("nvidia/geforce-rtx-3070")

mod, params = deserialize("unet")
mod["main"] = rewrite_attention(mod["main"])
mod["main"] = simplify_div(mod["main"])

ref = get_ref(mod, params, target, dev, inputs)

mod = run_opt_passes(mod)
mod = partition_for_cutlass(mod)
mod = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})(mod)

with target:
    mod = get_lower_passes(params)(mod)

out = get_result(mod, target, dev, inputs, time_eval=False)

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))
print(out)
