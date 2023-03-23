import tvm
from tvm.script import relax as R
from tvm.relax.dpl import *


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


# {VarPattern(x): x, Op(relax.call_dps_packed)(ExternFuncPattern(bias_add), *): lv4, Op(relax.call_dps_packed)(ExternFuncPattern(bias_add), *): lv1, Op(relax.call_dps_packed)(ExternFuncPattern(my_relu), *): lv2, Op(relax.call_dps_packed)(ExternFuncPattern(my_relu), *): lv5, Op(relax.call_dps_packed)(ExternFuncPattern(conv1x1), *): lv0, Op(relax.call_dps_packed)(ExternFuncPattern(conv1x1), *): lv3}

with PatternContext() as ctx:
    inp_pat = wildcard()
    Q_weight_pat = wildcard()
    K_weight_pat = wildcard()
    V_weight_pat = wildcard()

    matmul1 = is_op("relax.matmul")
    matmul2 = is_op("relax.matmul")
    matmul3 = is_op("relax.matmul")

    inp_pat.used_by(matmul1, 0)
    inp_pat.used_by(matmul2, 0)
    inp_pat.used_by(matmul3, 0)

    Q_weight_pat.only_used_by(matmul1, 1)
    K_weight_pat.only_used_by(matmul2, 1)
    V_weight_pat.only_used_by(matmul3, 1)

    dfb = QKV_proj["main"].body.blocks[0]
    out = ctx.match_dfb(dfb)
    print(out)
