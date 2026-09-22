import io, sys, time, soundfile as sf, openvino_genai as ov_genai
OUT = io.open("bench/verify.txt", "w", encoding="utf-8")
def P(*a): print(*a, file=OUT, flush=True)

M = r"D:\AI\Models"
asr  = M + r"\whisper-large-v3-turbo-int8-ov"
mt   = M + r"\Qwen3-4B-int4-ov"

audio, sr = sf.read("bench/speech.wav", dtype="float32")
t0 = time.perf_counter()
w = ov_genai.WhisperPipeline(asr, "GPU")
txt = w.generate(audio[:16000*10])
P(f"[ASR] load+infer {time.perf_counter()-t0:.2f}s")
P(f"  {str(txt)[:160]}")
del w

t0 = time.perf_counter()
p = ov_genai.LLMPipeline(mt, "GPU")
p.start_chat("You are a professional interpreter. Translate English to natural Chinese. Output ONLY the translation.")
r = p.generate("The scheduler performs a context switch. /no_think", max_new_tokens=128, do_sample=False)
p.finish_chat()
P(f"[MT ] load+infer {time.perf_counter()-t0:.2f}s")
P(f"  {str(r).replace('<think>','').replace('</think>','').strip()[:160]}")
OUT.close()
