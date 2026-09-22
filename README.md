# 同声传译 · Real-time EN→ZH Interpretation

英文语音 → 中文字幕，全部在本机 Intel Arc 140T 核显上推理，不联网、不上传音频。

```bash
python run.py --cli          # 终端模式
python run.py --gui          # 悬浮字幕窗
python run.py --devices      # 列出可用音频设备
```

---

## 硬件与实测数据

所有数字都是在本机实测的，不是估算。测量脚本保留在 `tools/` 和 `bench/`。

**平台**：Intel Core Ultra 9 285H · Arc 140T 核显（128 EU，带 XMX 矩阵引擎）· 31.5 GB LPDDR5x 共享内存

| 指标 | 核显 (GPU) | CPU | 结论 |
|---|---|---|---|
| FP16 算力 | **11.10 TFLOPS** | 0.56 TFLOPS | 核显强 20× |
| 有效内存带宽 | 51.0 GB/s | 72.8 GB/s | 共用同一条内存总线，核显**不占优** |
| Prefill（长提示词） | **1360–3875 tok/s** | 121–356 tok/s | 核显强 11× |
| Decode（逐词生成） | 15.5 tok/s | 相近 | 带宽瓶颈，核显无优势 |

**这套硬件的性格**：计算密集型负载（Whisper encoder、prompt prefill）核显碾压 CPU；
带宽瓶颈型负载（LLM 逐词解码）核显毫无优势。架构上的每个决策都是围绕这一点做的。

### 模型选型（实测对比后确定）

| 环节 | 选定模型 | 关键数据 |
|---|---|---|
| 语音识别 | **whisper-large-v3-turbo INT8** | RTF 0.026（38× 实时）；5s 分块 320ms，p95 343ms |
| 翻译 | **Qwen3-4B INT4** | 15.5 tok/s，TTFT 173ms |
| 备用翻译 | Qwen2.5-1.5B INT4 | 34.4 tok/s，积压时自动切换 |
| 断句 | Silero VAD v5 (ONNX) | 102 µs/帧，CPU |

**为什么 ASR 用最大的 turbo 而不是 small**：turbo 反而更快（2.48s vs 3.08s）。
它只有 4 层解码器（small 有 12 层），而庞大的 encoder 正好喂饱 XMX 单元。没有理由用更小的。

**为什么翻译用 4B 而不是 1.5B**：1.5B 把 `prefill` 译成「预读取」（那是 prefetch），
4B 正确译为「预填充」。技术场景下这种错误会直接误导听众。代价是慢 2.2×，通过流式输出抵消。

**为什么不用专用翻译模型**（NLLB / MADLAD / Hunyuan-MT）：它们都是句子级的，吃不进上下文。
实时口语翻译里代词指代和术语一致性靠的就是上下文，通用 LLM 在这点上完胜。
（另外 OpenVINO 官方也没有预转换版本，自己转要拖进整套 PyTorch。）

---

## 架构

```
T1  WASAPI 回调  ──► RingBuffer (float32 16kHz)        实时线程，不可阻塞
T2  vad-loop     ──► Silero VAD ──► Segmenter ──► 队列
T3  gpu-worker   ──► Whisper ──► 断句 ──► Qwen3 ──► EventBus     独占核显
T4  前端         ──► bus.drain()                       CLI 或 Qt 悬浮窗
```

### 核心决策：GPU 单线程独占

`tools/exp_contention.py` 实测了 ASR 和翻译并发抢核显的代价：

| | 单独 | 并发 | 劣化 |
|---|---|---|---|
| ASR p95 | 424 ms | 1337 ms | **3.15×** |
| MT 解码 | 15.18 tok/s | 9.88 tok/s | **1.54×** |
| | | 合计 **4.69**（阈值 2.0） | |

并发没有任何重叠收益——OpenVINO 把两者塞进同一个 GPU 命令队列，还要抢同一条 51 GB/s 内存总线。
**所以有且只有一个线程碰 GPU**，ASR 永远优先于翻译（转录过时会卡住整条流水线，而翻译晚一点只影响一行字幕）。

改 `runtime.gpu_concurrency` 之前请先重跑 E1。

### 前后端解耦

后端只往 `EventBus` 发事件，前端只 `drain()`。两边互不 import。
`app/pipeline.py` 里没有任何一行提到 UI —— 这就是为什么同一条流水线能同时驱动终端和 Qt 悬浮窗。

---

## 延迟预算

从**说话人最后一个音节**算起（不是从断句判定算起——那样会把最大的一项藏起来）：

| 环节 | 耗时 | 说明 |
|---|---|---|
| 静音等待（断句） | 550 ms | **最大项**，`vad.endpoint_silence_ms` 可调 |
| Whisper 识别 | ~360 ms | |
| 排队 + 翻译首字 | ~470 ms | 一段语音切出多句时，第 2 句要等第 1 句 |
| **首字出现** | **~1.38 s** (p50) | |
| 整句完成 | ~1.92 s (p50) | |

物理地板约 1.15 s。

### 为什么静音等待定在 550ms（实测扫参）

直觉上调低 `endpoint_silence_ms` 能直接省下同等延迟。实测否定了这一点：

| endpoint | 产生句数 | 首字 p50 | 排队+翻译 | 整句 p50 |
|---|---|---|---|---|
| **550 ms** | 18 | 1384 ms | 471 ms | 1918 ms |
| 350 ms | 23 | 1256 ms | 537 ms | 1670 ms |

省下 200ms 静音，首字只快了 128ms。原因是句子被切得更碎（同样音频多出 5 句），
排队时间反而涨了 66ms，吃掉大半收益，而且碎句的翻译质量更差。**550ms 是更好的平衡点。**

（注：测试音频由 TTS 合成，停顿结构比真人规整。真人语音停顿更多，两者差距可能不同。）

还有一个**急切断句**机制：如果部分转录已经以 `.?!` 结尾，250ms 静音就提交，不等满 550ms。
这是在不切碎句子的前提下省延迟——只在已经确认是完整句子时才提前。

---

## 背压策略

说得快的人会超过核显的串行处理能力。**一条迟到 30 秒的字幕比没有字幕更糟**，所以逐级降级：

| 积压 | 动作 |
|---|---|
| > 2 | 停止注入上下文和术语表（省 prefill） |
| ≥ 4 | 切到 Qwen2.5-1.5B（快 2.2×） |
| > 8 | 丢弃最旧的未翻译行，显示 `(… skipped)` |

---

## 磁盘占用：原地覆盖，不累积

模型放在 `D:\AI\Models`，**项目目录从不复制模型**，只按路径引用。

唯一会自己长大的是 OpenVINO 编译缓存 `.ovcache`（当前 3.1 GB）。`app/storage.py` 管着它：

- **运行时指纹校验**：OpenVINO 版本或 GPU 变了就整个清空——旧 blob 永远不会再被读取，留着纯属浪费。
- **LRU 修剪**：超过 `runtime.cache_budget_gb`（默认 8 GB）就按最久未访问顺序淘汰。

缓存值得留：冷启动 12.9s → 热启动 3.4s。

---

## 目录结构

```
run.py                  入口（--cli / --gui / --devices）
config.yaml             用户配置，代码里有全套默认值
glossary.tsv            术语表，保存即热重载

app/
  config.py             类型化配置，未知键会报错而不是静默忽略
  types.py              Segment / Line / 各类 UI 事件
  bus.py                前后端边界：有界队列，消费者慢了丢最旧的
  metrics.py            延迟统计与验收门槛
  storage.py            缓存指纹校验 + LRU 修剪
  console.py            Windows 控制台 UTF-8 修复（GBK 下打印中文会崩）
  audio/
    capture.py          AudioSource 抽象 + 工厂
    wasapi.py           PyAudioWPatch：系统回录 + 麦克风（主用）
    soundcard_src.py    soundcard 备用后端
    ring.py             环形缓冲，写端永不阻塞
    resample.py         → 16kHz 单声道
  vad.py                Silero VAD（含 v5 的 64 采样上下文坑）
  segmenter.py          断句状态机（纯函数，可单测）
  sentence_split.py     转录切分成翻译单元
  asr.py                Whisper 引擎 + 幻觉过滤
  mt.py                 Qwen3 引擎 + 持久 chat 会话
  think_filter.py       流式剥离 <think> 标签
  glossary.py / context.py
  gpu_worker.py         GPU 唯一所有者 + 背压
  pipeline.py           线程编排
  cli.py                终端前端
  ui/                   Qt 悬浮窗前端

tools/
  list_devices.py       列音频设备
  exp_contention.py     E1：GPU 争用实验
  exp_cache.py          E2：缓存收益实验
  replay.py             拿 wav 跑完整流水线，可复现
tests/                  54 个单测，不需要模型或声卡
bench/                  最初的硬件基准脚本（参考用，非生产代码）
```

---

## 两个踩过的坑

**Silero VAD v5 需要 64 采样上下文。** ONNX 图不接受裸的 512 采样帧，要把上一帧的最后
64 个采样拼在前面（共 576）。喂错了不报错，只是对着响亮的人声返回 0.0006。
`tests/test_vad.py` 锁死了这个行为。

**Qwen3 的 thinking 模式必须关。** 实测同一句话：开着 43.65 秒 / 674 token，
加 `/no_think` 后 1.61 秒 / 22 token。而且即使关了，模型仍会吐一对空的 `<think></think>`，
必须在**流式**过程中剥离——标签会跨 token 边界断开（`<thi` + `nk>`），
简单的 `str.replace` 会漏。见 `app/think_filter.py`。

---

## 验证

```bash
python -m pytest tests/ -q                    # 54 个单测，秒级
python tools/replay.py bench/speech.wav       # 端到端，按真实时间喂音频
python tools/exp_contention.py                # 改线程模型前必跑
```

`replay.py` 是回归测试的主力：同一个 wav 走和实时采集完全相同的代码路径，
输出延迟分解和验收判定。
