import time, openvino_genai as ov_genai
import os
para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "several engineers discuss cache coherency protocols and memory ordering "
        "semantics in modern multicore processors. ")
LONG = para * 110 + "\n\nSummarize the above text in one sentence."

for path, tag in [("phi35","Phi-3.5-mini 3.8B"), ("qwen15","Qwen2.5-1.5B")]:
    for dev in ["GPU","CPU"]:
        try:
            pipe = ov_genai.LLMPipeline(path, dev)
            pipe.generate("hi", max_new_tokens=8)
            st={"f":None}
            s=time.perf_counter()
            def cb(t):
                if st["f"] is None: st["f"]=time.perf_counter()
                return ov_genai.StreamingStatus.RUNNING
            pipe.generate(LONG, max_new_tokens=16, do_sample=False, streamer=cb)
            ttft=(st["f"]-s)
            ntok=len(pipe.get_tokenizer().encode(LONG).input_ids.data[0])
            print(f"{tag:20s} {dev:3s} | prompt {ntok:5d} tok | prefill {ttft*1000:8.1f} ms "
                  f"| {ntok/ttft:8.1f} tok/s", flush=True)
            del pipe
        except Exception as e:
            print(f"{tag} {dev} FAILED: {str(e)[:180]}", flush=True)
