import time, numpy as np, soundfile as sf, openvino_genai as ov_genai

audio, sr = sf.read("speech.wav", dtype="float32")
DUR = len(audio)/sr
print(f"audio: {DUR:.2f}s @ {sr}Hz\n")

MODELS = [("w-base","base"),("w-small","small"),("w-turbo","large-v3-turbo")]

print("=== FULL-CLIP (batch) ===")
for path, tag in MODELS:
    for dev in ["GPU","CPU"]:
        try:
            t0=time.perf_counter(); pipe=ov_genai.WhisperPipeline(path,dev)
            load=time.perf_counter()-t0
            pipe.generate(audio[:16000*3])                       # warmup
            t0=time.perf_counter(); r=pipe.generate(audio); el=time.perf_counter()-t0
            print(f"{tag:16s} {dev:3s} | load {load:5.1f}s | proc {el:6.2f}s | "
                  f"RTF {el/DUR:5.3f} | speed {DUR/el:6.1f}x realtime")
            del pipe
        except Exception as e: print(f"{tag} {dev} FAILED: {str(e)[:150]}")

print("\n=== STREAMING (5s chunks, GPU) ===")
CH=5*16000
for path, tag in MODELS:
    try:
        pipe=ov_genai.WhisperPipeline(path,"GPU"); pipe.generate(audio[:16000*3])
        lat=[]
        for i in range(0,len(audio)-CH,CH):
            t0=time.perf_counter(); pipe.generate(audio[i:i+CH]); lat.append(time.perf_counter()-t0)
        lat=np.array(lat)
        print(f"{tag:16s} | chunks {len(lat):2d} | mean {lat.mean()*1000:7.1f} ms | "
              f"p95 {np.percentile(lat,95)*1000:7.1f} ms | max {lat.max()*1000:7.1f} ms | "
              f"RTF {lat.mean()/5:5.3f}")
        del pipe
    except Exception as e: print(f"{tag} FAILED: {str(e)[:150]}")
