import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite_call, rewrite_bindings, is_op, wildcard, is_const, PatternContext
from tvm.script import relax as R
from tvm.relax.transform import ConvertLayout, Normalize, ToMixedPrecision, EliminateCommonSubexpr, CanonicalizeBindings


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

        for i, matmul_pat in enumerate(matmul_patterns):
            width = weights[i].struct_info.shape[1]
            bound_var = matchings[matmul_pat]
            replacements[bound_var] = R.strided_slice(matmul, axes=[slice_axis], begin=[begin], end=[begin + width])
            begin += width

        return replacements

    return rewrite_bindings(ctx, rewriter, f)


def rewrite_qkv_proj(f):
    return combine_parallel_matmul(f, 3)


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


with open("unet.json", "r") as fi:
    mod_tmp = tvm.ir.load_json(fi.read())

mod = tvm.IRModule()
mod["main"] = mod_tmp["unet"]
mod = EliminateCommonSubexpr()(mod)
mod = CanonicalizeBindings()(mod)
mod["main"] = rewrite_attention(mod["main"])
mod["main"] = combine_parallel_matmul(mod["main"], 32)
mod["main"] = combine_parallel_matmul(mod["main"], 22, slice_axis=1)
mod["main"] = rewrite_qkv_proj(mod["main"])
mod["main"] = simplify_div(mod["main"])
mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(mod)
# mod = ToMixedPrecision(out_dtype="float16")(mod)

mod = partition_for_cutlass(mod)
print(mod)
