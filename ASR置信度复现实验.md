# ASR 置信度复现实验

## 当前结论

本次先复现“置信度提取 → 聚合 → 后校准”的安全最小链路，没有改变在线服务的默认行为。

当前 Qwen3-ASR 流式服务默认调用 `qwen_asr` 的 `streaming_transcribe()`，该接口只更新
`state.language` 和 `state.text`。本次增加了可选的 `--confidence-logprobs K` 适配器，
在不改变默认文本解码的前提下保留 vLLM 生成结果中的 `logprobs`，并让
`/stream/chunk` 和 `/stream/finish` 返回实验性 confidence 字段。

## 已复用的思路

- NVIDIA NeMo：token/word/frame confidence，支持 `mean`、`min`、`max`、`prod` 聚合以及
  max-probability 和 entropy 计算。
- `whisper-timestamped`：从 Whisper 的 token/cross-attention 信息生成词级和句级分数。
- Adaptive GER：用多候选 ASR 和风险控制决定是否、以及向精修模型传入多少候选。

这些项目不能直接作为 Qwen 流式服务的即插即用模块，但可以复用数据字段和聚合方式。

## 本地实现

新增 [system/asr_confidence.py](system/asr_confidence.py)：

1. 读取 `confidence`、`asr_confidence`；
2. 读取 NeMo 风格 `word_confidence`、`token_confidence`、`frame_confidence`；
3. 读取 Whisper 风格 `avg_logprob`；
4. 区分已校准概率和未校准的 logprob 派生分数；
5. 提供无第三方依赖的 temperature scaling 和 ECE 计算。

特别地，`avg_logprob` 会标记为 `calibrated=False`，不能直接拿来和门控阈值 `0.8` 或
`0.92` 比较。

Qwen 流式服务启动示例：

```bash
python -m system.qwen_asr_stream_server \
  --model /path/to/Qwen3-ASR-1.7B \
  --confidence-logprobs 5 \
  --vad silero \
  --vad-model models/silero_vad.onnx \
  --vad-min-silence 0.7 \
  --vad-min-speech 0.25 \
  --vad-preroll 0.35 \
  --port 8766
```

当前服务返回的字段包括 `confidence`、`confidence_mean_logprob`、
`confidence_min_probability`、`confidence_token_count` 和
`confidence_calibrated`。实测静音输入也能返回一组分数，因此必须结合 VAD、文本内容和
校准集使用，不能单独把高分当作正确识别。

当前启动脚本同时启用了会话级 Silero VAD。VAD 会先判断是否有人声，并使用 350ms
前置缓冲和 0.7 秒静音保持时间；RMS 仅作为 VAD 关闭时的兜底。纯静音 PCM 块会在
Qwen 之前被拦截，并返回 `silence_skipped=true`；这一步是活动检测，不等同于 ASR
confidence。运行 Qwen 服务的环境需要安装 `sherpa-onnx`，并准备
`models/silero_vad.onnx`（可运行 `bash system/download_vad.sh models` 下载）。

## 最小使用示例

```python
from system.asr_confidence import extract_confidence, TemperatureCalibrator

evidence = extract_confidence({"word_confidence": [0.91, 0.84, 0.95]})
print(evidence.public_dict())

calibrator = TemperatureCalibrator.fit(
    [0.95, 0.90, 0.70, 0.20],
    [1, 0, 1, 0],
)
calibrated = calibrator.transform(evidence.value or 0.0)
```

## 验证结果

```text
python -m unittest discover -s tests -v
Ran 81 tests ... OK
```

新增的置信度测试覆盖：显式 confidence、NeMo 词级聚合、Whisper avg_logprob 的未校准标记、
temperature scaling 和 ECE。

## 接入当前 Qwen 门控

已在 Qwen 流式解码层保留 vLLM `RequestOutput`，而不能只读取生成文本：

```text
vLLM SamplingParams(logprobs=K)
        ↓
保存每个生成 token 的 logprob
        ↓
按 token/句聚合
        ↓
用项目开发集做校准
        ↓
只在 final/已提交窗口把 calibrated confidence 传给 RefinementGate
```

默认启动参数只打开分数采集，不宣称已经校准。完成开发集校准后，再加上
`--confidence-temperature T --confidence-calibrated`，并将校准后的分数用于门控；
在此之前，建议把当前结果当作 shadow/实验信号，避免仅凭 raw score 跳过实体精修。
