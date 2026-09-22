import time, openvino_genai as ov_genai

PROMPT = ("You are a helpful assistant. Explain in detail how a modern CPU "
          "pipeline works, covering fetch, decode, execute, memory and writeback "
          "stages, and why branch prediction matters for performance.")
MAXNEW = 200

def run(path, dev, tag):
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(path, dev)
    load = time.perf_counter() - t0
    pipe.generate("hi", max_new_tokens=8)                      # warm up

    state = {"n": 0, "first": None}
    start = time.perf_counter()
    def cb(tok):
        state["n"] += 1
        if state["first"] is None: state["first"] = time.perf_counter()
        return ov_genai.StreamingStatus.RUNNING
    pipe.generate(PROMPT, max_new_tokens=MAXNEW, do_sample=False, streamer=cb)
    end = time.perf_counter()

    ttft = (state["first"] - start) * 1000
    dec  = (state["n"] - 1) / (end - state["first"]) if state["n"] > 1 else 0
    print(f"{tag:24s} {dev:3s} | load {load:5.1f}s | TTFT {ttft:7.1f} ms | "
          f"decode {dec:6.2f} tok/s | total {end-start:5.1f}s / {state['n']} tok", flush=True)
    del pipe

for path, tag in [("phi35","Phi-3.5-mini 3.8B"), ("qwen15","Qwen2.5-1.5B")]:
    for dev in ["GPU", "CPU"]:
        try: run(path, dev, tag)
        except Exception as e: print(f"{tag} {dev} FAILED: {str(e)[:200]}", flush=True)
