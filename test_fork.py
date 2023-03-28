import tvm
from tvm.script import relax as R
from tvm.relax.dpl import *


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
                lv0 = R.call_dps_packed("conv1x1", (x, w0), R.Tensor((32, 32), dtype="float32"))
                lv1 = R.call_dps_packed("bias_add", (lv0, bias0), R.Tensor((32, 32), dtype="float32"))
                lv2 = R.call_dps_packed("my_relu", (lv1), R.Tensor((32, 32), dtype="float32"))
                lv3 = R.call_dps_packed("conv1x1", (x, w1), R.Tensor((32, 32), dtype="float32"))
                lv4 = R.call_dps_packed("bias_add", (lv3, bias1), R.Tensor((32, 32), dtype="float32"))
                lv5 = R.call_dps_packed("my_relu", (lv4), R.Tensor((32, 32), dtype="float32"))
                lv6 = R.call_dps_packed("concat", (lv2, lv5), R.Tensor((32, 64), dtype="float32"))
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
            x: R.Tensor((2, 1024, 640), "float32"),
            w0: R.Tensor((640, 640), "float32"),
            w1: R.Tensor((640, 640), "float32"),
            w2: R.Tensor((640, 640),"float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(x, w1)
                lv2 = R.matmul(x, w2)
                out = (lv0, lv1, lv2)
                R.output(out)
            return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        is_op("relax.matmul")(inp_pat, Q_weight_pat)
        is_op("relax.matmul")(inp_pat, K_weight_pat)
        is_op("relax.matmul")(inp_pat, V_weight_pat)

        dfb = QKV_proj["main"].body.blocks[0]
        out = ctx.match_dfb(dfb)

        assert out[Q_weight_pat].name_hint == "w0"
        assert out[K_weight_pat].name_hint == "w1"
        assert out[V_weight_pat].name_hint == "w2"


def test_CBR_x2_op():
    @tvm.script.ir_module
    class CBRx2:
        @R.function
        def main(
            x: R.Tensor((1, 3, 32, 32), "float32"),
            w0: R.Tensor((32, 3, 1, 1), "float32"),
            bias0: R.Tensor((1, 32, 1, 1), "float32"),
            w1: R.Tensor((32, 3, 1, 1), "float32"),
            bias1: R.Tensor((1, 32, 1, 1), "float32"),
        ) -> R.Tensor:
            # R.TensorRT's CBR Optimization Pattern
            #     input
            #     /   \
            #  cbr0   cbr1
            #     \   /
            #     concat
            with R.dataflow():
                lv0 = R.nn.conv2d(x, w0, padding=(1, 1))
                lv1 = lv0 + bias0
                lv2 = R.nn.relu(lv1)
                lv3 = R.nn.conv2d(x, w1, padding=(1, 1))
                lv4 = lv3 + bias1
                lv5 = R.nn.relu(lv4)
                lv6 = R.concat((lv2, lv5))
                R.output(lv6)
            return lv6

    with PatternContext() as ctx:
        inp = wildcard()
        weight1 = wildcard()
        weight2 = wildcard()
        bias1_pat = wildcard()
        bias2_pat = wildcard()

        conv1 = is_op("relax.nn.conv2d")(inp, weight1)
        bias1 = is_op("relax.add")(conv1, bias1_pat)
        relu1 = is_op("relax.nn.relu")(bias1)

        conv2 = is_op("relax.nn.conv2d")(inp, weight2)
        bias2 = is_op("relax.add")(conv2, bias2_pat)
        relu2 = is_op("relax.nn.relu")(bias2)

        dfb = CBRx2["main"].body.blocks[0]
        out = ctx.match_dfb(dfb)

        print(out)
        # print(out[weight1])


test_qkv()
# test_CBR_x2()
