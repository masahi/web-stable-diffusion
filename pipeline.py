import numpy as np

import tvm
from tvm import relax


def get_result(ex, dev, inputs, time_eval=False):
    vm = relax.VirtualMachine(ex, dev, profile=True)
    out = vm["main"](*inputs)

    if time_eval:
        print(vm.profile("main", *inputs))
        # vm.set_input("main", *inputs)
        # print(vm.time_evaluator("invoke_stateful", dev, repeat=50)("main"))

    return out.numpy()


def test(model):
    so_name = "{}.so".format(model)

    target = tvm.target.Target("nvidia/geforce-rtx-3070")

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
                np.random.randint(low=0, high=1000, size=(1, 77)).astype("int32"), dev
            )
        ]

    ex = tvm.runtime.load_module(so_name)

    out = get_result(ex, dev, inputs, time_eval=True)


test("vae")
