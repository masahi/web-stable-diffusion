import numpy as np

import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite_call, is_op, wildcard, is_const
from tvm.script import relax as R
from tvm.relax.transform.tuning_api import Trace


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

    matmul1 = is_op("relax.matmul")(Q_3D, is_op("relax.permute_dims")(K_3D))
    multiply = is_op("relax.multiply")(matmul1, is_const())
    softmax = is_op("relax.nn.softmax")(multiply)
    matmul2 = is_op("relax.matmul")(softmax, V_3D)

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


def run_opt_passes(mod, params=None):
    if params:
        return tvm.transform.Sequential(
            [
                relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
                relax.transform.BindParams("main", params),
                relax.transform.FoldConstant(),
                relax.transform.ToMixedPrecision(out_dtype="float16"),
            ]
        )(mod)

    return tvm.transform.Sequential(
            [
                relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
                relax.transform.ToMixedPrecision(out_dtype="float16"),
            ]
        )(mod)


def run_lower_passes(mod, target, tune=False):
    passes = [relax.pipeline.get_pipeline()]

    if "cuda" in target.kind.name:
        if not tune:
            passes.append(tir.transform.DefaultGPUSchedule())
        else:
            work_dir = "work"
            with target:
                # passes.append(relax.transform.MetaScheduleTuneIRMod(
                #     params={},
                #     work_dir=work_dir,
                #     max_trials_global=500,
                # ))
                passes.append(relax.transform.MetaScheduleApplyDatabase(work_dir))
                passes.append(tir.transform.DefaultGPUSchedule())

    # with target, tvm.transform.PassContext(trace=Trace(mod), opt_level=0):
    with target, tvm.transform.PassContext(opt_level=3):
        return tvm.transform.Sequential(passes)(mod)


def get_result(ex, dev, inputs, params=None, time_eval=False):
    vm = relax.VirtualMachine(ex, dev, profile=True)

    # if params:
    #     inputs.append(params.values())

    out = vm["main"](*inputs)

    if time_eval:
        print(vm.profile("main", *inputs))
        # vm.set_input("main", *inputs)
        # print(vm.time_evaluator("invoke_stateful", dev, repeat=50)("main"))

    return out.numpy()


def get_ref(mod, params, target, dev, inputs, bind_params=True):
    passes = []

    if bind_params:
        passes += [relax.transform.BindParams("main", params),
                   relax.transform.FoldConstant()]

    passes.append(relax.transform.ToMixedPrecision(out_dtype="float16"))

    mod = tvm.transform.Sequential(passes)(mod)

    mod = run_lower_passes(mod, target)
    ex = relax.build(mod, target)

    if bind_params:
        return get_result(ex, dev, inputs)

    return get_result(ex, dev, inputs, params=params)


bind_params = True

model = "clip"

if bind_params:
    so_name = "{}.so".format(model)
else:
    so_name = "{}_no_params.so".format(model)

# target = tvm.target.Target("nvidia/geforce-rtx-3070")
target = tvm.target.Target("llvm")

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

mod, params = deserialize(model)

ref = get_ref(mod, params, target, dev, inputs, bind_params=bind_params)

mod["main"] = rewrite_attention(mod["main"])
mod["main"] = simplify_div(mod["main"])

if bind_params:
    mod = run_opt_passes(mod, params)
else:
    mod = run_opt_passes(mod)

# mod = partition_for_cutlass(mod)
# mod = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})(mod)

mod = run_lower_passes(mod, target, tune=True)

ex = relax.build(mod, target)
ex.export_library(so_name)

if bind_params:
    out = get_result(ex, dev, inputs, time_eval=False)
else:
    out = get_result(ex, dev, inputs, params=params, time_eval=False)

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))
