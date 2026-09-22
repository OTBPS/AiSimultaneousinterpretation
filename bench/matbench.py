import time, sys, numpy as np, openvino as ov
import openvino.opset13 as ops

N = 1024          # matrix dim
CHAIN = 32        # chained matmuls -> compute bound
core = ov.Core()

def build(dtype):
    x = ops.parameter([N, N], dtype, name="x")
    cur = x
    for i in range(CHAIN):
        w = ops.constant(np.random.randn(N, N).astype(dtype) * 0.02)
        cur = ops.matmul(cur, w, False, False)
    return ov.Model([cur], [x], "chain")

FLOPS_PER_INFER = 2.0 * N * N * N * CHAIN

for dev, dt in [("CPU", np.float32), ("GPU", np.float16), ("NPU", np.float16)]:
    try:
        m = build(dt)
        t0 = time.perf_counter()
        cm = core.compile_model(m, dev)
        comp = time.perf_counter() - t0
        req = cm.create_infer_request()
        inp = np.random.randn(N, N).astype(dt)
        for _ in range(2):                      # warmup
            req.infer({0: inp})
        runs, t0 = 8, time.perf_counter()
        for _ in range(runs):
            req.infer({0: inp})
        dur = (time.perf_counter() - t0) / runs
        print(f"{dev:4s} {str(np.dtype(dt)):8s} compile={comp:6.2f}s  "
              f"latency={dur*1000:8.2f}ms  {FLOPS_PER_INFER/dur/1e12:7.2f} TFLOPS")
    except Exception as e:
        print(f"{dev:4s} FAILED: {str(e)[:160]}")
