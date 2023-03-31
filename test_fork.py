import tvm
from tvm.script import relax as R
from tvm.relax.dpl import *
from tvm import relax


def test_CBR_x2():
    @tvm.script.ir_module
    class CBRx2:
        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w0: R.Tensor((1, 1), "float32"),
            bias0: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((1, 1), "float32"),
            bias1: R.Tensor((32, 32), "float32"),
        ) -> R.Tensor:
            # R.TensorRT's CBR Optimization Pattern
            #     input
            #     /   \
            #  cbr0   cbr1
            #     \   /
            #     concat
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "conv1x1", (x, w0), R.Tensor((32, 32), dtype="float32")
                )
                lv1 = R.call_dps_packed(
                    "bias_add", (lv0, bias0), R.Tensor((32, 32), dtype="float32")
                )
                lv2 = R.call_dps_packed(
                    "my_relu", (lv1), R.Tensor((32, 32), dtype="float32")
                )
                lv3 = R.call_dps_packed(
                    "conv1x1", (x, w1), R.Tensor((32, 32), dtype="float32")
                )
                lv4 = R.call_dps_packed(
                    "bias_add", (lv3, bias1), R.Tensor((32, 32), dtype="float32")
                )
                lv5 = R.call_dps_packed(
                    "my_relu", (lv4), R.Tensor((32, 32), dtype="float32")
                )
                lv6 = R.call_dps_packed(
                    "concat", (lv2, lv5), R.Tensor((32, 64), dtype="float32")
                )
                R.output(lv6)
            return lv6

    with PatternContext() as ctx:
        conv = is_call_dps_packed("conv1x1")
        bias = is_call_dps_packed("bias_add")
        relu = is_call_dps_packed("my_relu")
        conv.used_by(bias, 0)
        bias.used_by(relu, 0)

        cbr0 = conv
        cbr1 = cbr0.dup()

        weight1 = wildcard()
        weight2 = wildcard()

        weight1.used_by(cbr0, 1)
        weight2.used_by(cbr1, 1)

        is_var("x").fork_to(cbr0, cbr1)
        dfb = CBRx2["main"].body.blocks[0]
        out = ctx.match_dfb(dfb)

        print(out)
        # print(out[weight1])


def test_qkv():
    @tvm.script.ir_module
    class QKV_proj:
        @R.function
        def main(
            x1: R.Tensor((2, 1024, 640), "float32"),
            x2: R.Tensor((2, 1024, 640), "float32"),
            w0: R.Tensor((640, 640), "float32"),
            w1: R.Tensor((640, 640), "float32"),
            w2: R.Tensor((640, 640), "float32"),
            w3: R.Tensor((640, 640), "float32"),
            w4: R.Tensor((640, 640), "float32"),
            w5: R.Tensor((640, 640), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.matmul(x1, w0)
                lv1 = R.matmul(x1, w1)
                lv2 = R.matmul(x1, w2)
                lv3 = R.matmul(x2, w3)
                lv4 = R.matmul(x2, w4)
                lv5 = R.matmul(x2, w5)
                out = (lv0, lv1, lv2, lv3, lv4, lv5)
                R.output(out)
            return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        def rewriter(matchings):
            inp = matchings[inp_pat]
            Q_weight = matchings[Q_weight_pat]
            K_weight = matchings[K_weight_pat]
            V_weight = matchings[V_weight_pat]
            width = Q_weight.struct_info.shape[1]

            concat = R.concat([Q_weight, K_weight, V_weight], axis=1)
            matmul = R.matmul(inp, concat)
            Q = R.strided_slice(matmul, axes=[2], begin=[0], end=[width])
            K = R.strided_slice(matmul, axes=[2], begin=[width], end=[width * 2])
            V = R.strided_slice(matmul, axes=[2], begin=[width * 2], end=[width * 3])

            return {matchings[matmul1]: Q, matchings[matmul2]: K, matchings[matmul3]: V}

        print(rewrite_bindings(ctx, rewriter, QKV_proj["main"]))


test_qkv()
