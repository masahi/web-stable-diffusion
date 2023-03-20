import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.dpl import rewrite, is_op, wildcard, is_const
from tvm.script import relax as R
from tvm.relax.transform import ConvertLayout, Normalize, ToMixedPrecision


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

    return rewrite(pattern, callback, f)


def rewrite_qkv_proj(f):
    def matmul_transposed_reshape(tensor, weight_transposed):
        # lv51: R.Tensor((320, 320), dtype="float32") = R.permute_dims(unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q_weight, axes=None)
        # lv52: R.Tensor((2, 4096, 320), dtype="float32") = R.matmul(lv50, lv51, out_dtype="float32")
        # lv_1: R.Tensor((2, 4096, 8, 40), dtype="float32") = R.reshape(lv52, R.shape([2, 4096, 8, 40]))
        return is_op("relax.reshape")(is_op("relax.matmul")(tensor, weight_transposed), wildcard())

    inp_pat = wildcard()
    Q_weight_pat = wildcard()
    K_weight_pat = wildcard()
    V_weight_pat = wildcard()

    Q = matmul_transposed_reshape(inp_pat, Q_weight_pat)
    K = matmul_transposed_reshape(inp_pat, K_weight_pat)
    V = matmul_transposed_reshape(inp_pat, V_weight_pat)

    pattern = is_op("relax.nn.attention")(Q, K, V)

    def callback(_, matchings):
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

    return rewrite(pattern, callback, f)


def rewrite_qkv_proj2(f):
    def matmul_transposed_reshape(tensor, weight_transposed):
        # lv51: R.Tensor((320, 320), dtype="float32") = R.permute_dims(unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q_weight, axes=None)
        # lv52: R.Tensor((2, 4096, 320), dtype="float32") = R.matmul(lv50, lv51, out_dtype="float32")
        # lv_1: R.Tensor((2, 4096, 8, 40), dtype="float32") = R.reshape(lv52, R.shape([2, 4096, 8, 40]))
        return is_op("relax.reshape")(is_op("relax.matmul")(tensor, weight_transposed), wildcard())

    inp_pat = wildcard()
    Q_weight_pat = wildcard()
    K_weight_pat = wildcard()
    V_weight_pat = wildcard()

    Q = matmul_transposed_reshape(inp_pat, Q_weight_pat)
    K = matmul_transposed_reshape(inp_pat, K_weight_pat)
    V = matmul_transposed_reshape(inp_pat, V_weight_pat)

    pattern = is_op("relax.nn.attention")(Q, K, V)

    def callback(_, matchings):
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

    return rewrite(pattern, callback, f)


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

    return rewrite(pattern, callback, f)


with open("unet.json", "r") as fi:
    mod_tmp = tvm.ir.load_json(fi.read())

mod = tvm.IRModule()
mod["main"] = rewrite_attention(mod_tmp["unet"])
mod["main"] = rewrite_qkv_proj(mod["main"])
mod["main"] = simplify_div(mod["main"])
mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(mod)
mod = Normalize()(mod)
# mod = ToMixedPrecision(out_dtype="float16")(mod)

mod = partition_for_cutlass(mod)
print(mod)
