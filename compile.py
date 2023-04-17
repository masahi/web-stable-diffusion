import numpy as np

import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite_call, is_op, wildcard, is_const, PatternContext, rewrite_bindings
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


def rewrite_qkv_proj(f):
    def matmul_transposed_reshape(tensor, weight_transposed):
        return is_op("relax.reshape")(is_op("relax.matmul")(tensor, weight_transposed), wildcard())

    inp_pat = wildcard()
    Q_weight_pat = wildcard()
    K_weight_pat = wildcard()
    V_weight_pat = wildcard()

    Q = matmul_transposed_reshape(inp_pat, Q_weight_pat)
    K = matmul_transposed_reshape(inp_pat, K_weight_pat)
    V = matmul_transposed_reshape(inp_pat, V_weight_pat)

    pattern = is_op("relax.nn.attention")(Q, K, V)

    def rewriter(_, matchings):
        inp = matchings[inp_pat]
        Q_weight = matchings[Q_weight_pat]
        K_weight = matchings[K_weight_pat]
        V_weight = matchings[V_weight_pat]

        width = Q_weight.struct_info.shape[1]
        weight_concat = R.concat([Q_weight, K_weight, V_weight], axis=1)
        matmul = R.matmul(inp, weight_concat)
        Q = R.strided_slice(matmul, axes=[2], begin=[0], end=[width])
        K = R.strided_slice(matmul, axes=[2], begin=[width], end=[width*2])
        V = R.strided_slice(matmul, axes=[2], begin=[width*2], end=[width*3])

        num_head = 8
        hidden = width // num_head
        seq_len = inp.struct_info.shape[1]

        Q = R.reshape(Q, R.shape([2, seq_len, num_head, hidden]))
        K = R.reshape(K, R.shape([2, seq_len, num_head, hidden]))
        V = R.reshape(V, R.shape([2, seq_len, num_head, hidden]))

        return R.nn.attention(Q, K, V)

    return rewrite_call(pattern, rewriter, f)


def combine_parallel_matmul(f, num_branches, slice_axis=2):
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

        begin = 0
        replacements = {}

        sections = []
        ind = 0
        for i, matmul_pat in enumerate(matmul_patterns[:-1]):
            width = weights[i].struct_info.shape[1]
            ind += width
            sections.append(int(ind))

        chunks = R.split(matmul, sections, slice_axis)

        for i, matmul_pat in enumerate(matmul_patterns):
            bound_var = matchings[matmul_pat]
            replacements[bound_var] = chunks[i]

        return replacements

    return rewrite_bindings(ctx, rewriter, f)


def run_opt_passes(mod):
    return tvm.transform.Sequential(
        [
            # relax.transform.EliminateCommonSubexpr(),
            # relax.transform.CanonicalizeBindings(),
            # relax.transform.CombineParallelMatmul(),
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

mod, params = deserialize("unet")

mod["main"] = rewrite_attention(mod["main"])
mod["main"] = simplify_div(mod["main"])

mod = relax.transform.EliminateCommonSubexpr()(mod)
mod = relax.transform.CanonicalizeBindings()(mod)
mod = relax.transform.CombineParallelMatmul()(mod)

# mod["main"] = combine_parallel_matmul(mod["main"], 32)
# mod["main"] = combine_parallel_matmul(mod["main"], 22, slice_axis=1)
# mod["main"] = combine_parallel_matmul(mod["main"], 3)
# mod["main"] = combine_parallel_matmul(mod["main"], 2)

print(mod)

# mod = run_opt_passes(mod)

# mod = partition_for_cutlass(mod)
# mod = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})(mod)

# target = tvm.target.Target("nvidia/geforce-rtx-3070")
# with target:
#     mod = get_lower_passes(params)(mod)

# dev = tvm.device("cuda", 0)
# inp_0 = tvm.nd.array(np.random.randn(1, 4, 64, 64).astype("float32"), dev)
# inp_1 = tvm.nd.array(np.array(1, "int32"), dev)
# inp_2 = tvm.nd.array(np.random.randn(2, 77, 768).astype("float32"), dev)
# inputs = [inp_0, inp_1, inp_2]

# ref = get_ref(mod, params, target, dev, inputs)
# out = get_result(mod, target, dev, inputs, time_eval=False)

# print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))
# print(out)
