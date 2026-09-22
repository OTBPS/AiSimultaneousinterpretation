import time, io, numpy as np, openvino_genai as ov_genai
OUT = io.open("results.txt", "w", encoding="utf-8")
def P(*a):
    print(*a, file=OUT, flush=True)

SYS = "You are a professional simultaneous interpreter. Translate the user's English into natural, fluent Chinese. Output ONLY the translation, no explanation."
CASES = [
 ("口语省略", "Yeah, no, I mean — we could ship it Friday, but honestly? Gonna be rough."),
 ("技术术语", "The scheduler preempts the running thread when a higher-priority task becomes runnable, then performs a context switch."),
 ("长难句", "Although the initial benchmarks suggested that the new allocator would reduce fragmentation, subsequent testing under sustained load revealed that it actually increased peak memory usage by roughly twelve percent."),
 ("上下文代词", "Context: We evaluated the Arc 140T integrated GPU against a discrete card.\nTranslate this: It turned out to be much better at prefill, though it lagged on decode."),
 ("习语", "Let's not boil the ocean here — just get a rough cut in front of the team by Thursday."),
]

def stream(pipe, msg, maxnew):
    st = {"f": None, "n": 0}; out = []
    def cb(t):
        st["n"] += 1
        if st["f"] is None: st["f"] = time.perf_counter()
        out.append(t); return ov_genai.StreamingStatus.RUNNING
    t0 = time.perf_counter()
    pipe.generate(msg, max_new_tokens=maxnew, do_sample=False, streamer=cb)
    e = time.perf_counter()
    tps = (st["n"]-1)/(e-st["f"]) if st["n"] > 1 else 0
    return "".join(out), (st["f"]-t0)*1000, e-t0, st["n"], tps

P("="*70); P("### 稳态解码速度 (400 token 长输出)"); P("="*70)
for path, tag, nt in [("qwen15","Qwen2.5-1.5B",False), ("q3-4b","Qwen3-4B (/no_think)",True)]:
    pipe = ov_genai.LLMPipeline(path, "GPU"); pipe.generate("hi", max_new_tokens=4)
    _,_,_,n,tps = stream(pipe, "Write a 400-word essay about memory bandwidth." + (" /no_think" if nt else ""), 400)
    P(f"{tag:26s} {tps:6.2f} tok/s   ({n} tok)")
    del pipe

for path, tag, nt in [("qwen15","Qwen2.5-1.5B-Instruct INT4",False), ("q3-4b","Qwen3-4B INT4 (/no_think)",True)]:
    pipe = ov_genai.LLMPipeline(path, "GPU")
    pipe.start_chat(SYS); pipe.generate("hi", max_new_tokens=4); pipe.finish_chat()
    P(f"\n{'='*70}\n### {tag}\n{'='*70}")
    lats = []
    for name, src in CASES:
        pipe.start_chat(SYS)
        txt, ttft, tot, n, tps = stream(pipe, src + (" /no_think" if nt else ""), 256)
        pipe.finish_chat()
        lats.append((ttft, tot, tps))
        txt = txt.replace("<think>","").replace("</think>","").strip().replace("\n"," ")
        P(f"\n[{name}]  TTFT {ttft:6.0f}ms | 总计 {tot:5.2f}s | {n:3d} tok | {tps:5.1f} tok/s")
        P(f"  → {txt[:300]}")
    a = np.array(lats)
    P(f"\n-- 平均: TTFT {a[:,0].mean():.0f}ms | 总延迟 {a[:,1].mean():.2f}s | {a[:,2].mean():.1f} tok/s --")
    del pipe

P("\n" + "="*70); P("### thinking 模式未关闭时的代价 (Qwen3-4B)"); P("="*70)
pipe = ov_genai.LLMPipeline("q3-4b", "GPU"); pipe.generate("hi", max_new_tokens=4)
pipe.start_chat(SYS)
txt, ttft, tot, n, tps = stream(pipe, CASES[0][1], 1024)   # 不加 /no_think
pipe.finish_chat()
P(f"同一句口语省略,thinking 开启: 总延迟 {tot:.2f}s | {n} tok")
P(f"  → {txt.strip()[:400]}")
OUT.close()
