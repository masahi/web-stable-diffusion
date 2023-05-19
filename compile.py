import numpy as np
import pickle
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import (
    rewrite_call,
    is_op,
    wildcard,
    is_const,
    PatternContext,
    rewrite_bindings,
)
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


def simplify_stride_slice(f):
    inp_pat = wildcard()
    pattern = is_op("relax.strided_slice")(inp_pat)

    def is_nop(v, begin, end, strides):
        shape = v.struct_info.shape
        for i in range(len(shape)):
            if begin[i] != 0 or end[i] != shape[i] or strides[i] != 1:
                return False
        return True

    def callback(orig, matchings):
        inp = matchings[inp_pat]
        if is_nop(inp, orig.attrs.begin, orig.attrs.end, orig.attrs.strides):
            return inp
        return orig

    return rewrite_call(pattern, callback, f)


def combine_parallel_matmul(f, num_branches):
    with PatternContext() as ctx:
        inp_pat = wildcard()

        weight_patterns = []
        matmul_patterns = []

        for _ in range(num_branches):
            w_pat = wildcard()
            weight_patterns.append(w_pat)
            matmul_patterns.append(is_op("relax.matmul")(inp_pat, w_pat))

    def rewriter(matchings):
        inp = matchings[inp_pat]

        weights = [matchings[w_pat] for w_pat in weight_patterns]
        concat = R.concat(weights, axis=1)
        matmul = R.matmul(inp, concat)

        replacements = {}

        sections = []
        ind = 0
        for i, matmul_pat in enumerate(matmul_patterns[:-1]):
            width = weights[i].struct_info.shape[1]
            ind += width
            sections.append(int(ind))

        if len(inp.struct_info.shape) == 3:
            slice_axis = 2
        elif len(inp.struct_info.shape) == 2:
            slice_axis = 1
        else:
            assert False

        chunks = R.split(matmul, sections, slice_axis)

        for i, matmul_pat in enumerate(matmul_patterns):
            bound_var = matchings[matmul_pat]
            replacements[bound_var] = chunks[i]

        return replacements

    return rewrite_bindings(ctx, rewriter, f)


def get_rewrite_pass(combine_matmul=False):
    @tvm.transform.module_pass(opt_level=0)
    def rewrite_passes(mod, _):
        mod["main"] = rewrite_attention(mod["main"])
        mod["main"] = simplify_div(mod["main"])
        mod["main"] = simplify_stride_slice(mod["main"])

        if combine_matmul:
            mod["main"] = combine_parallel_matmul(mod["main"], 46)
            mod["main"] = combine_parallel_matmul(mod["main"], 22)
            mod["main"] = combine_parallel_matmul(mod["main"], 3)

        return mod

    return rewrite_passes


def run_opt_passes(mod, params=None, fp16_input_names=None, combine_matmul=False):
    passes = [
        relax.transform.EliminateCommonSubexpr(),
        relax.transform.CanonicalizeBindings(),
        get_rewrite_pass(combine_matmul),
        relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]}),
    ]

    if params:
        passes += [
            relax.transform.BindParams("main", params),
            relax.transform.FoldConstant(),
            relax.transform.ToMixedPrecision(out_dtype="float16"),
        ]
    else:
        passes += [
            relax.transform.FoldConstant(),
            relax.transform.ToMixedPrecision(
                out_dtype="float16", fp16_input_names=fp16_input_names
            ),
        ]

    return tvm.transform.Sequential(passes)(mod)


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
                #     max_trials_global=1500,
                #     # max_trials_per_task=50,
                #     # op_names=["group_norm"]
                # ))
                passes.append(relax.transform.MetaScheduleApplyDatabase(work_dir))
                passes.append(tir.transform.DefaultGPUSchedule())

    with target, tvm.transform.PassContext(opt_level=3):
        return tvm.transform.Sequential(passes)(mod)


def get_result(ex, dev, inputs, params_fp16=None):
    vm = relax.VirtualMachine(ex, dev, profile=True)

    if params_fp16:
        params_gpu = [p.copyto(dev) for p in params_fp16]
        params_transformed = vm["main_transform_params"](params_gpu)
        inputs.append(params_transformed)

    out = vm["main"](*inputs)
    return out.numpy()


def add_params_to_input(inputs, params, param_names, dev):
    new_inputs = [inp for inp in inputs]

    for p in param_names:
        new_inputs.append(tvm.nd.array(params[p].numpy(), dev))
    return new_inputs


def get_ref(mod, params, target, dev, inputs, bind_params=True):
    passes = []

    if bind_params:
        passes += [
            relax.transform.BindParams("main", params),
            relax.transform.FoldConstant(),
        ]

    passes.append(relax.transform.ToMixedPrecision(out_dtype="float16"))

    mod = tvm.transform.Sequential(passes)(mod)

    mod = run_lower_passes(mod, target)
    ex = relax.build(mod, target)

    if not bind_params:
        param_names = [p.name_hint for p in mod["main"].params[len(inputs) :]]
        inputs = add_params_to_input(inputs, params, param_names, dev)

    return get_result(ex, dev, inputs)


bind_params = False
verify = False
combine_matmul = True

model = "unet"
# hidden_dim = 1024 # for v2.1
hidden_dim = 768  # for v1.5

if bind_params:
    so_name = "{}.so".format(model)
else:
    so_name = "{}_no_params.so".format(model)

target = tvm.target.Target("nvidia/geforce-rtx-3070")
# target = tvm.target.Target("llvm")

dev = tvm.device(target.kind.name, 0)
inp_0 = tvm.nd.array(np.random.randn(2, 4, 64, 64).astype("float32"), dev)
inp_1 = tvm.nd.array(np.array(1, "int32"), dev)
inp_2 = tvm.nd.array(np.random.randn(2, 77, hidden_dim).astype("float32"), dev)

if model == "unet":
    controlnet_cond = tvm.nd.array(
        np.random.randn(2, 3, 512, 512).astype("float32"), dev
    )
    inputs = [inp_0, inp_1, inp_2, controlnet_cond]
elif model == "vae":
    inputs = [inp_0]
else:
    inputs = [
        tvm.nd.array(
            np.random.randint(low=0, high=1000, size=(1, 77)).astype("int64"), dev
        )
    ]

mod, params = deserialize(model)

if not bind_params:
    param_names = [p.name_hint for p in mod["main"].params[len(inputs) :]]
    params_fp16 = {}

    for i, name in enumerate(param_names):
        params_fp16[name + f"_{i}"] = tvm.nd.array(
            params[name].numpy().astype("float16")
        )

    tvm.runtime.save_param_dict_to_file(params_fp16, "{}_fp16.params".format(model))

if verify:
    ref = get_ref(mod, params, target, dev, inputs, bind_params=bind_params)

if bind_params:
    mod = run_opt_passes(mod, params, combine_matmul=combine_matmul)
else:
    fp16_input_names = [p.name_hint for p in mod["main"].params[len(inputs) :]]
    mod = run_opt_passes(
        mod, fp16_input_names=fp16_input_names, combine_matmul=combine_matmul
    )

if "cuda" in target.kind.name:
    mod = partition_for_cutlass(mod)
    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}}
    )(mod)

mod = run_lower_passes(mod, target, tune=True)

if not bind_params:
    mod = relax.transform.LiftTransformParams()(mod)

with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": False}):
    ex = relax.build(mod, target)

ex.export_library(so_name)

if verify:
    if bind_params:
        out = get_result(ex, dev, inputs)
    else:
        params = [params_fp16[name + f"_{i}"] for i, name in enumerate(param_names)]
        out = get_result(ex, dev, inputs, params)

    print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))
