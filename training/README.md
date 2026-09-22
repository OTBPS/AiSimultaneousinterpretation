# 蒸馏：14B 教师 → 4B 学生

把台式机 4090 上 Qwen3-14B 的翻译质量，转移到笔记本上跑的 Qwen3-4B，**速度不变**。

## 为什么是蒸馏，不是直接部署 14B

实测推算（基于本机 Qwen3-4B INT4 = 2.29 GB → 15.5 tok/s，解码是带宽瓶颈）：

| 模型 | INT4 体积 | 解码 | 5 秒语音的 GPU 占用率 |
|---|---|---|---|
| Qwen3-4B | 2.3 GB | 15.5 tok/s | ~27% |
| Qwen3-8B | 4.9 GB | ~7.6 tok/s | ~55% |
| Qwen3-14B | 9.7 GB | ~4.4 tok/s | **~96%** |

14B 在这台笔记本上占用率逼近 100%，说话人稍快就积压，背压会把它降级到 1.5B —
**等于花 14B 的代价跑 1.5B**。蒸馏则让 4B 保持 15.5 tok/s 不变。

## 两台机器怎么配合

**不需要任何网络连接。** 全流程离线，只有一个 2.3 GB 的文件夹需要拷贝。

```
台式机 4090                                        笔记本
──────────────────────────────                     ─────────────────
corpus.txt  (你的领域英文语料，单语即可)
      │  1_generate.py      Qwen3-14B 批量翻译
      ▼
train.jsonl  (英中对照)
      │  2_train_lora.py    QLoRA 微调 Qwen3-4B
      ▼
lora_adapter/  (~100 MB)
      │  3_export_openvino.py   合并 + 导出 + INT4 量化
      ▼
Qwen3-4B-ft-int4-ov/  (2.3 GB) ──── 拷贝 ────► D:\AI\Models\
                                                改 config.yaml 的 mt.model
```

笔记本上**不需要装 PyTorch**。台式机上**不需要跑 OpenVINO 推理**（只在第 3 步用它导出，纯 CPU）。

## 关键：英文侧应当来自真实音频，不是书面文本

中文那一侧由 14B 生成，所以不需要人工翻译的平行语料 —— 这是最大的优势。

但英文侧的来源很讲究。**学生上线后收到的从来不是书面英文，而是 Whisper 的输出**，
本项目实测中它长这样：

```
"...contention when multiple accelerations"      ← 把 accelerators 听错了
"The operators access shared system memories."   ← 句子被切错
"copies between stages."                         ← 硬切产生的碎片
"and third, the hiring plan for..."              ← 小写开头，断句瑕疵
```

拿干净书面英文训练，等于教模型应对一个它永远不会遇到的分布 ——
这是微调「训练时好看、上线后无效」最常见的原因。

所以用 `0_prepare_corpus.py` 在笔记本上转录真实音频，
**它复用生产环境同一套 ASR、VAD 和断句逻辑**，产出的文本分布与生产完全一致，
包括其中的缺陷。不要清洗这些噪声。

```bash
python training/0_prepare_corpus.py --audio "D:\talks" --out corpus.txt
```

实测 22× 实时（比纯 ASR 的 38× 慢，因为加了 VAD 和断句）：一小时音频约 2.7 分钟。

音频来源：你的课程录像、行业讲座、会议录音、播客。目标 **1 万～5 万句**。
与你实际使用场景越接近越好，授权问题自行把关。

## 交给 Claude Code 自动跑？

见 `DESKTOP_AGENT.md`，里面有可直接粘贴的任务简报和验收门槛。
一句话版本：**agent 负责执行和监控，语料选择和最终质量判定必须你来做。**

## 台式机环境

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl datasets accelerate bitsandbytes
pip install vllm                      # 可选，教师生成快 10 倍以上
pip install optimum[openvino] nncf    # 第 3 步导出用
```

## 运行

```bash
# 1. 教师生成（14B，~20k 句，vLLM 批处理约 30 分钟）
python 1_generate.py --corpus corpus.txt --out train.jsonl \
                     --teacher Qwen/Qwen3-14B --max-samples 20000

# 2. QLoRA 微调（4090 约 1~3 小时）
python 2_train_lora.py --data train.jsonl --base Qwen/Qwen3-4B \
                       --out lora_adapter --epochs 2

# 3. 合并 + 导出 OpenVINO INT4（约 15 分钟）
python 3_export_openvino.py --base Qwen/Qwen3-4B --adapter lora_adapter \
                            --out Qwen3-4B-ft-int4-ov
```

显存占用（24 GB 卡）：

| 步骤 | 占用 | 备注 |
|---|---|---|
| 1 教师生成 14B | ~16 GB | AWQ/GPTQ 4bit；bf16 需 28 GB 放不下 |
| 2 QLoRA 4B | ~10 GB | 4bit 基座 + LoRA + 梯度检查点 |
| 2 LoRA (bf16 基座) | ~16 GB | 质量略好，也能放下 |
| 3 导出 | CPU 为主 | 合并时峰值内存约 16 GB（系统内存） |

## 回到笔记本后

```yaml
# config.yaml
mt:
  model: 'D:\AI\Models\Qwen3-4B-ft-int4-ov'
```

然后**必须跑评测对比**，否则你不知道有没有变好：

```bash
python tools/eval_translation.py \
  --models D:\AI\Models\Qwen3-4B-int4-ov D:\AI\Models\Qwen3-4B-ft-int4-ov
```

基线已经存在 `eval_baseline.json`（12/12 通过，平均 1230 ms/句）。

## 诚实的预期

**能学到**：术语译法、句式风格、语气、格式规范、领域习惯表达。

**学不到**：14B 多出来的知识和推理能力。4B 不会因为学了 14B 的输出就变成 14B。
典型情况能追回师生差距的三到六成。

**可能变差**：过拟合到教师的口癖；INT4 量化会吃掉一部分微调增益。
所以第 2 步要留验证集，第 3 步导出后要重新跑评测。

## 先别急着训练

在投入一整晚之前，先确认这件事值得做：

1. **跑 `tools/eval_translation.py` 对比 4B 和 14B**（14B 可以在笔记本上慢慢跑，
   只有 12 句，不在乎速度）。**如果差距不明显，蒸馏就不值得。**
2. **实测真人语音的识别准确率**。如果错误主要来自 Whisper 听错，
   换多好的翻译模型都没用 —— 错误在上游就产生了。
3. **先填术语表**。实测发现它改变了 12 句里的 9 句，成本为零且随时可撤。
   但注意：术语表只适合精度重要的技术词，习语交给模型自己处理更自然
   （"boil the ocean" 强制成「好高骛远」反而不如模型自己的「别一上来就搞得太大太全」）。
