import time, numpy as np, openvino as ov
import openvino.opset13 as ops
core = ov.Core()
N, CHAIN = 4096, 24          # weights: 24 x 4096x4096 fp16 = 768 MB read per pass
def build(dt):
    x = ops.parameter([1, N], dt, name="x"); cur = x
    for _ in range(CHAIN):
        cur = ops.matmul(cur, ops.constant((np.random.randn(N,N)*0.02).astype(dt)), False, False)
    return ov.Model([cur], [x], "mv")
BYTES = CHAIN * N * N * 2    # fp16 weight bytes streamed per inference
for dev in ["GPU", "CPU"]:
    dt = np.float16 if dev == "GPU" else np.float32
    b = BYTES * (2 if dev == "CPU" else 1)
    m = build(dt); cm = core.compile_model(m, dev); req = cm.create_infer_request()
    inp = np.random.randn(1, N).astype(dt)
    for _ in range(3): req.infer({0: inp})
    r, t0 = 15, time.perf_counter()
    for _ in range(r): req.infer({0: inp})
    d = (time.perf_counter()-t0)/r
    print(f"{dev}: {d*1000:7.2f} ms/pass  effective bandwidth = {b/d/1e9:6.1f} GB/s")
