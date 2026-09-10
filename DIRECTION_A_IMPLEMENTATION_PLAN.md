# 方向 A：基于置信度门控的可控流式 ASR 精修方案

## 1. 研究目标

本方案以当前 AgenticASR 项目为基础，研究“何时精修、精修哪一段、如何保证不改错”，最终实现一个同时支持 Whisper 和 Qwen3-ASR 的低延迟、可控、可回退流式语音识别系统。

系统需要同时维护两种输出：

- `verbatim_text`：尽量忠实保留说话内容，包括必要的重复、停顿、自我修正和口语现象。
- `intended_text`：在不改变最终意图的前提下，去除无意义重复和口头语，补充标点并规范数字、日期和术语。

核心研究问题如下：

1. 如何判断一段 ASR 文本是否真的需要调用 Refiner？
2. 如何只修改当前不稳定片段，避免历史文本持续跳动？
3. 如何防止 Refiner 修改正确的人名、数字、术语和代码？
4. 如何在降低 Refiner 调用次数和延迟的同时保持文本质量？
5. 方法能否跨 Whisper、Qwen3-ASR 和不同领域泛化？

建议论文题目：

> 面向真实场景的意图保持型低延迟流式语音识别与可控文本精修方法研究

## 2. 总体架构

完整链路设计为：

```text
浏览器/音频文件
    ↓
ASR Adapter（Whisper / Qwen3-ASR）
    ↓
统一 ASR Hypothesis 协议
    ↓
Hypothesis Tracker（稳定性、修订率、可提交前缀）
    ↓
Feature Extractor（置信度、停顿、不流畅、实体风险）
    ↓
Refinement Gate（KEEP / DEFER / REFINE）
    ↓
Active Span Planner（只选择允许修改的后缀）
    ↓
Protected-span Masker（数字、实体、代码占位保护）
    ↓
Structured Refiner（结构化输出）
    ↓
Validator（实体、语义、编辑范围、一致性检查）
    ↓
接受精修结果或回退到 ASR 原文
    ↓
WebSocket 双输出 + JSONL 实验日志
```

系统内部采用以下状态机：

```text
COLLECTING → CANDIDATE → REFINING → VALIDATING → COMMITTED
                  ↘ DEFER                    ↘ FALLBACK_RAW
```

## 3. 目录与模块规划

建议新增以下文件：

```text
system/
├── contracts.py             # 统一 ASR、门控、精修数据结构
├── hypothesis_tracker.py    # 增量假设稳定性与修订分析
├── feature_extractor.py     # 门控特征抽取
├── gating.py                # 规则门控和学习型门控
├── span_planner.py          # 动态可编辑区域选择
├── protection.py            # 数字、实体、术语占位保护
├── validators.py            # 精修结果安全校验
├── refinement_engine.py     # 串联门控、精修、验证和回退
└── session_metrics.py       # 流式延迟、稳定性和资源指标

experiments/scripts/
├── replay_stream.py         # 使用已保存事件复现实验
├── evaluate_dual_reference.py
├── evaluate_streaming.py
├── train_gate.py
├── calibrate_confidence.py
└── build_dual_reference.py

configs/
├── research.yaml
├── gate_rule.yaml
└── protected_terms.txt
```

需要修改的现有文件：

- `system/web_app.py`：接入 RefinementEngine，使用异步调用，输出门控和验证信息。
- `system/qwen_asr_stream_server.py`：返回统一 ASR Hypothesis，尽可能暴露 token logprob。
- `system/whisper_backend.py`：返回 token 分数、语言和可用时间戳。
- `system/whisper_stream_server.py`：限制滚动音频窗口，避免无限累计和重复计算。
- `system/chunking.py`：从固定字符切块升级为稳定性和停顿驱动的动态切块。
- `system/refiner.py`：支持结构化返回值、只读上下文和可编辑目标文本。
- `system/web/index.html`：分别显示逐字稿、意图稿、门控状态和实时延迟。

## 4. 第一步：建立统一 ASR 输出协议

### 4.1 数据结构

在 `system/contracts.py` 中定义：

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class ASRToken:
    text: str
    start_ms: float | None = None
    end_ms: float | None = None
    logprob: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class ASRHypothesis:
    session_id: str
    sequence_id: int
    text: str
    language: str | None
    audio_end_ms: float
    received_at_ms: float
    is_final: bool
    tokens: list[ASRToken] = field(default_factory=list)
    mean_logprob: float | None = None
    confidence: float | None = None
    no_speech_prob: float | None = None
    backend: str = "unknown"
```

所有字段允许缺失，但不得伪造置信度。模型不提供某项信息时使用 `None`，并增加 `feature_missing` 标记供门控模型处理。

### 4.2 Whisper Adapter

在 `WhisperBackend.transcribe()` 之外增加 `transcribe_detailed()`：

1. 调用 `generate()` 时启用 `return_dict_in_generate=True` 和 `output_scores=True`。
2. 使用 `compute_transition_scores()` 得到生成 token 的 log probability。
3. 计算：
   - token 平均 logprob；
   - token 最小 logprob；
   -低置信 token 比例；
   - 每秒 token 数。
4. 如果当前 Transformers 接口无法稳定返回词级时间戳，在线阶段将时间戳留空；离线评测可以使用 WhisperX 对齐，不应把离线对齐延迟计入在线系统延迟。

当前 Whisper 流式服务会反复识别全部累计音频，需要改为：

- 仅保存最近 `30 s` 音频作为活动窗口；
- 保存已经提交的稳定文本前缀；
- 每次只对“左侧上下文 3–5 s + 新音频”重新识别；
- 通过最长公共前缀或 token 对齐合并历史结果；
- 会话结束时允许执行一次完整 final pass，单独记录其延迟。

### 4.3 Qwen3-ASR Adapter

Qwen3-ASR 服务返回统一字段：

```json
{
  "session_id": "...",
  "sequence_id": 12,
  "text": "我觉得这个方案",
  "language": "Chinese",
  "audio_end_ms": 4800,
  "is_final": false,
  "tokens": [],
  "mean_logprob": null,
  "confidence": null,
  "backend": "qwen3-asr"
}
```

如果 Qwen 推理接口支持 logprobs，则填入真实 logprob；如果不支持，门控主要使用文本稳定性、停顿、重复和历史修订特征。这样可以检验门控是否真正跨 ASR 后端泛化。

## 5. 第二步：增量假设稳定性分析

在 `system/hypothesis_tracker.py` 中维护最近若干次 ASR 假设：

```python
class HypothesisTracker:
    def update(self, hypothesis: ASRHypothesis) -> StabilityState:
        ...
```

每次更新计算：

- `lcp_ratio`：当前文本与上一版本最长公共前缀占比；
- `revision_ratio`：两次文本的归一化编辑距离；
- `unchanged_updates`：某个 token 连续多少次未变化；
- `tail_age_ms`：尾部文本已保持不变的时间；
- `chars_per_second` 或 `tokens_per_second`；
- 最近一秒内的假设更新次数；
- 是否出现句末标点；
- VAD 静音持续时间。

初始提交规则：

1. token 连续 `3` 次更新未变化，或结束时间距离当前音频超过 `800 ms`，标记为 stable；
2. stable token 可以进入只读上下文；
3. 最近 `12–30` 个字符保留为活动后缀；
4. final 事件到来时提交全部剩余文本；
5. 如果已提交前缀被 ASR 修改，不直接抛异常，而是生成 `rollback_event` 并限制最大回滚长度。

这些数值是启动默认值，最终通过开发集搜索，而不是直接作为论文结论。

## 6. 第三步：实现精修门控

门控有三个输出：

- `KEEP`：当前文本无需精修，直接使用 ASR 原文；
- `DEFER`：信息不足，等待更多上下文；
- `REFINE`：调用 Refiner。

### 6.1 第一版：规则门控

在 `system/feature_extractor.py` 中提取四类特征。

ASR 特征：

- mean/min logprob；
- token entropy；
- 低置信 token 比例；
- no-speech probability；
- 语言和后端类型。

流式特征：

- `lcp_ratio`；
- `revision_ratio`；
- `unchanged_updates`；
- 静音长度；
- 活动窗口时长；
- 距离上次 Refiner 调用的时间。

文本特征：

- 中文口头语数量，如“嗯、呃、那个、就是”；
- 重复 unigram、bigram 和短语比例；
- “不对、我是说、应该是”等自我修正词；
- 句末标点是否完整；
- 数字、日期、英文缩写、代码符号数量；
- 中英切换次数；
- 异常字符或过长无标点片段。

风险特征：

- 人名、机构名和领域词数量；
- 数字是否连续变化；
- 当前假设是否修改了已经稳定的实体；
- Refiner 上一次是否被 Validator 拒绝。

初始门控规则：

```text
p_need < 0.25                  → KEEP
0.25 ≤ p_need < 0.65          → DEFER
p_need ≥ 0.65 且尾部已稳定     → REFINE
final 且 p_need ≥ 0.45         → REFINE
final 且 p_need < 0.45         → KEEP
```

规则分数可以先按以下形式实现：

```text
risk =
    0.20 × ASR低置信度
  + 0.20 × 假设修订率
  + 0.25 × 不流畅概率
  + 0.15 × 标点缺失概率
  + 0.20 × 自我修正概率
```

需要增加以下节流机制：

- 两次在线 Refiner 调用间隔默认不少于 `600 ms`；
- 同一文本不得重复精修；
- 新请求到达时，尚未开始的旧请求取消；
- 已经完成但 sequence_id 过期的结果直接丢弃；
- 同一会话只保留最新候选，避免当前 `refiner_lock` 导致请求排队。

### 6.2 第二版：学习型门控

数据积累后，使用开发集训练二分类模型：

```text
y = 1：精修后 intended CER 明显下降，且未破坏保护实体
y = 0：精修无收益、改坏文本或不需要修改
```

实现顺序：

1. Logistic Regression：作为可解释基线；
2. LightGBM 或小型 MLP：融合标量特征；
3. 可选：增加小型中文文本编码器的句向量；
4. 在开发集上做温度缩放或 isotonic regression 校准概率；
5. 搜索阈值，使质量、延迟和调用成本达到最佳平衡。

训练目标除分类损失外，应在实验分析中定义效用：

```text
Utility = intended质量提升
          - λ × Refiner延迟
          - μ × 语义漂移风险
          - ν × 调用成本
```

## 7. 第四步：动态活动窗口

当前 `StreamingRefinementSession` 使用固定三个 chunk。新方案将输入分成：

```text
[已经提交，不可修改的历史]
[只读左上下文]
[允许修改的 active span]
```

动态窗口规则：

1. stable prefix 永久提交，默认不允许 Refiner 修改；
2. 从当前尾部向左找到最近的停顿、句末标点或稳定边界；
3. active span 最短约 `8` 个字符，最长约 `80` 个字符或 `6 s` 音频；
4. 向左附加 `12–24` 个字符作为只读上下文；
5. 出现自我修正词时，窗口向左扩展到被修正内容；
6. 出现说话人切换或长静音时强制关闭当前窗口；
7. 无标点长句达到最大长度时强制形成候选，但只有稳定后才精修。

Refiner 接收的内容示例：

```json
{
  "readonly_context": "今天会议主要讨论项目进度。",
  "editable_text": "然后嗯下周下周一我们开始测试",
  "language": "Chinese",
  "protected_spans": ["下周一"]
}
```

Refiner 只能返回 `editable_text` 的替代文本，服务端负责与历史前缀拼接。

## 8. 第五步：受约束的 Refiner

### 8.1 结构化输出

将当前自由文本返回改为：

```json
{
  "decision": "edit",
  "text": "然后下周一我们开始测试。",
  "reason_codes": ["FILLER_REMOVAL", "REPETITION_REMOVAL", "PUNCTUATION"]
}
```

如果模型认为无需修改：

```json
{
  "decision": "keep",
  "text": "",
  "reason_codes": []
}
```

服务端不信任模型生成的字符偏移。实际 edit operations 由服务端使用字符级或 token 级最短编辑对齐计算。

### 8.2 Prompt 约束

系统提示词至少包含：

```text
1. 只允许修改 editable_text，readonly_context 仅供理解。
2. 不得增加原音频中没有的信息。
3. 保留最终意图，不得总结或扩写。
4. 保护占位符必须原样保留且只能出现一次。
5. 不确定时返回 keep。
6. 仅输出符合给定 Schema 的 JSON。
```

当前 `<KEY>...</KEY>` 不应再拼接到用户可见文本中。关键词和实体应作为独立 metadata 保存，避免污染 CER/WER。

### 8.3 实体与数字保护

在进入 Refiner 前识别：

- 整数、小数、百分比和货币；
- 日期和时间；
- 电话号码、邮箱、网址；
- 英文缩写和代码标识符；
- 用户词典中的人名、机构名和领域术语。

将其替换为占位符：

```text
原文：预算大概是 12.5 万，联系人是 Zhang Wei
掩码：预算大概是 ⟦P0⟧ 万，联系人是 ⟦P1⟧
```

Refiner 返回后恢复占位符。任何占位符丢失、重复、重排或被修改，都直接拒绝此次精修并回退原文。

## 9. 第六步：Validator 与安全回退

在 `system/validators.py` 中实现以下检查器：

### 9.1 SchemaValidator

- 输出必须是合法 JSON；
- `decision` 只能为 `keep` 或 `edit`；
- `text` 必须是字符串；
- 解析失败最多重试一次，仍失败则回退原文。

### 9.2 ProtectedSpanValidator

- 每个保护占位符必须出现且只出现一次；
- 恢复后数字、日期、实体和代码必须与原文一致；
- 保护失败立即拒绝。

### 9.3 EditScopeValidator

- 只允许修改 active span；
- 不允许修改 readonly context；
- 计算字符级编辑距离和新增内容比例；
- 删除发生在已检测的不流畅片段内时给予较高容忍度；
- 对正常内容的大段删除或新增执行拒绝。

初始阈值可以设置为：

```text
normalized_edit_distance ≤ 0.35
added_content_ratio ≤ 0.15
length_ratio ∈ [0.55, 1.30]
```

这些阈值必须在开发集上调整。对于含大量重复和口头语的样本，长度下界需要动态放宽。

### 9.4 SemanticValidator

在线快速检查：

- 比较关键词覆盖率；
- 检查否定词是否增加、删除或反转；
- 检查主谓宾核心词是否大规模改变；
- 可选使用轻量多语言句向量模型计算余弦相似度。

离线严格检查：

- 使用更强语义模型或 LLM Judge；
- 重点检查否定、数字、实体和事实新增；
- Judge 只用于评测，不作为唯一论文指标。

最终接受条件：

```text
accepted = schema_ok
           and protected_spans_ok
           and edit_scope_ok
           and semantic_ok
```

任一条件失败时返回 ASR 原文，并记录 `reject_reason`。

## 10. 第七步：改造 WebSocket 实时服务

当前 `system/web_app.py` 在每次文本变化时同步调用 Refiner，并通过一个线程锁串行执行。建议改为：

1. `_stream_request()` 改成异步 HTTP 客户端；
2. 每个 WebSocket 会话创建独立 `RefinementEngine`；
3. ASR 更新先写入 tracker，再由 gate 决定是否创建 Refiner 任务；
4. Refiner 任务携带 `sequence_id`；
5. 返回结果时检查 sequence_id，旧结果不得覆盖新结果；
6. 使用 latest-only 队列，避免请求积压；
7. 把 ASR 延迟、gate 延迟、Refiner 延迟和 Validator 延迟分别记录。

WebSocket 更新协议：

```json
{
  "event": "update",
  "sequence_id": 12,
  "verbatim_text": "然后嗯下周下周一我们开始测试",
  "intended_text": "然后下周一我们开始测试。",
  "active_span": "然后嗯下周下周一我们开始测试",
  "gate": {
    "decision": "REFINE",
    "probability": 0.83,
    "reasons": ["FILLER", "REPETITION"]
  },
  "validator": {
    "accepted": true,
    "reasons": []
  },
  "latency_ms": {
    "asr": 138,
    "gate": 2,
    "refiner": 214,
    "validator": 3
  }
}
```

前端显示：

- 原始逐字稿；
- 精修后的意图稿；
- 当前模式：KEEP、等待上下文或正在精修；
- Refiner 调用次数；
- 当前延迟和 p95 延迟；
- 精修被拒绝时的原因，仅在调试模式显示。

## 11. 第八步：双参考数据集

每条样本至少包含：

```json
{
  "sample_id": "meeting_000123",
  "audio_path": "...",
  "speaker_id": "spk_018",
  "domain": "meeting",
  "verbatim_reference": "我觉得嗯下周下周一开始测试",
  "intended_reference": "我觉得下周一开始测试。",
  "disfluency_spans": [
    {"text": "嗯", "type": "FILLER"},
    {"text": "下周", "type": "REPETITION"}
  ],
  "protected_spans": [
    {"text": "下周一", "type": "DATE"}
  ],
  "language": "zh",
  "noise_condition": "office"
}
```

### 11.1 数据规模

可行的研究生阶段配置：

- 起步集：`1–2 h`，用于打通标注和评测流程；
- 正式集：`5–10 h` 真实对话；
- 测试集至少 `1 h`，不得用于阈值调整；
- 按说话人划分训练、开发、测试集，不能随机按句划分；
- 测试集由两名标注者独立标注并处理分歧。

建议优先选择一个明确场景，例如实验室会议或课堂讨论。后续再增加噪声、中英混合和不同口音作为泛化实验。

### 11.2 标注规范

`verbatim_reference`：

- 保留实际说出的内容；
- 保留重复、自我修正和有意义停顿；
- 不擅自改变语序；
- 标点按照听感边界补充。

`intended_reference`：

- 去除不承载信息的口头语；
- 合并机械重复；
- 自我修正保留说话人的最终选择；
- 不总结、不补充背景信息；
- 数字、实体、否定关系必须与音频一致。

## 12. 第九步：评测指标

### 12.1 准确性与可控性

- `CER_verbatim`：ASR 原文对逐字参考；
- `CER_intended`：精修结果对意图参考；
- WER/MER：用于英文和中英混合；
- Disfluency Precision、Recall、F1；
- Entity Accuracy；
- Number Accuracy；
- 意图保持准确率；
- Refiner 错改率；
- Unsupported Addition Rate；
- Validator 接受率和误拒率。

不能只报告精修后的 CER，因为删除全部口语内容可能降低 intended CER，却破坏逐字忠实度。逐字稿和意图稿必须分别评测。

### 12.2 流式指标

- Time to First Token；
- Time to First Stable Token；
- Finalization Latency；
- Endpoint Delay；
- p50/p95 更新延迟；
- Real-Time Factor；
- 每分钟 Refiner 调用次数；
- Normalized Revision Rate；
- 稳定前缀长度；
- 用户可见文本闪烁次数；
- 峰值 GPU 显存和平均 GPU 利用率。

### 12.3 统计方法

- 对主要指标报告 bootstrap 95% 置信区间；
- 同一样本上的两个系统使用配对统计检验；
- 分领域、说话人、句长、噪声和中英混合情况报告分组结果；
- 阈值只能在开发集确定，测试集只运行一次最终配置。

## 13. 第十步：实验矩阵

### 13.1 基线系统

| 编号 | 系统 | 作用 |
|---|---|---|
| B1 | Whisper only | Whisper 原始基线 |
| B2 | Qwen3-ASR only | 中文/多语言原始基线 |
| B3 | ASR + always refine | 当前无条件精修方案 |
| B4 | ASR + final-only refine | 只在结束时调用 Refiner |
| B5 | ASR + rule gate | 规则门控 |
| B6 | ASR + learned gate | 学习型门控 |
| B7 | ASR + gate + validator | 增加安全校验 |
| Proposed | gate + dynamic span + protected refiner + validator | 完整方法 |

如条件允许，可以加入 CrisperWhisper、FormalASR 或其他端到端正式文本模型作为外部对比。

### 13.2 消融实验

完整方法分别移除：

- ASR confidence；
- hypothesis stability；
- disfluency features；
- dynamic span；
- protected-span masking；
- semantic validator；
- learned gate，仅保留规则；
- 双阈值 DEFER 机制；
- latest-only 调度。

每项消融同时观察质量、延迟、调用次数和错改率，不能只比较 CER。

### 13.3 泛化实验

- Whisper → Qwen3-ASR 跨后端；
- 干净语音 → 噪声语音；
- 普通话 → 中英混合；
- 短句 → 长对话；
- 训练领域 → 未见领域；
- 在线模式 → 离线模式。

## 14. 日志与可复现实验

每次会话写入独立 JSONL 事件流：

```json
{
  "session_id": "...",
  "sequence_id": 12,
  "audio_end_ms": 4800,
  "asr": {
    "backend": "qwen3-asr",
    "text": "...",
    "confidence": null,
    "latency_ms": 138
  },
  "stability": {
    "lcp_ratio": 0.91,
    "revision_ratio": 0.07,
    "unchanged_updates": 3
  },
  "gate": {
    "decision": "REFINE",
    "probability": 0.83,
    "latency_ms": 2
  },
  "refiner": {
    "input": "...",
    "output": "...",
    "latency_ms": 214
  },
  "validator": {
    "accepted": true,
    "reasons": []
  }
}
```

必须实现 `replay_stream.py`，让门控、窗口和 Validator 可以基于同一批 ASR 事件重复实验，无需每次重新运行 ASR。这能保证不同方法使用完全相同的输入，也能显著节约 GPU 时间。

配置文件中固定：

- 模型名称和 commit/hash；
- Transformers、Torch 和 CUDA 版本；
- 所有门控阈值；
- 随机种子；
- 数据集划分；
- Prompt 版本；
- Refiner decoding 参数；
- 机器和 GPU 信息。

## 15. 实施顺序与验收标准

### 阶段 0：冻结当前基线，预计 1 周

工作：

- 固定当前 Whisper、Qwen3-ASR 和 Refiner 版本；
- 保存一组可重复的音频样本；
- 记录原始 WER/CER、Refiner 延迟和调用次数；
- 给现有 JSONL 输出增加配置签名。

验收：同一配置重复运行得到一致结果，基线数据可以一条命令生成。

### 阶段 1：统一协议和事件日志，预计 2 周

工作：

- 实现 `contracts.py`；
- 修改两个 ASR 服务；
- 实现完整事件日志；
- 实现流式 replay。

验收：Whisper 和 Qwen3-ASR 产生同一 Schema，replay 结果与在线记录一致。

### 阶段 2：规则门控，预计 2–3 周

工作：

- 实现稳定性特征和文本特征；
- 实现 KEEP/DEFER/REFINE；
- 增加节流、去重和过期结果丢弃；
- 与 always-refine、final-only 比较。

目标：在 intended 质量基本不下降的条件下，将 Refiner 调用次数降低约 50%。这是研究目标，需要由实验验证，不作为预设结论。

### 阶段 3：受约束 Refiner 和 Validator，预计 3 周

工作：

- 结构化 JSON 输出；
- 数字、实体和代码占位保护；
- 编辑范围和语义校验；
- 自动回退机制。

目标：保护实体准确率不低于原始 ASR，并显著降低 Refiner 错改率。

### 阶段 4：动态活动窗口，预计 3–4 周

工作：

- stable prefix 和 active span；
- VAD、标点和修订率联合确定边界；
- 限制 Whisper 重识别窗口；
- 完成窗口大小消融实验。

目标：降低 p95 更新延迟和用户可见文本修订率。

### 阶段 5：双参考数据和学习型门控，预计 1–2 个月

工作：

- 完成双参考标注；
- 生成 gate 训练标签；
- 训练并校准门控模型；
- 完成跨 ASR、跨领域实验。

目标：学习型门控在相同调用预算下优于规则门控。

### 阶段 6：论文实验，预计 1 个月

工作：

- 运行全部基线和消融；
- 统计显著性和置信区间；
- 完成错误类型分析；
- 挑选成功与失败案例；
- 整理系统图、延迟—质量曲线和论文表格。

## 16. 最小可行版本（MVP）

为了避免一开始改动过大，第一版只实现：

1. 统一 ASR Hypothesis；
2. LCP、revision ratio 和不流畅词特征；
3. 规则 KEEP/DEFER/REFINE；
4. `600 ms` 调用节流和 latest-only 调度；
5. 数字、日期、英文缩写占位保护；
6. JSON 结构化 Refiner；
7. always-refine、final-only、rule-gate 三组对比；
8. 调用次数、p95 延迟、intended CER 和数字准确率四个主要指标。

完成 MVP 后再加入学习型门控、语义 Validator、动态窗口和双参考完整标注。

## 17. 预期论文贡献

最终论文可以形成四点贡献：

1. 提出跨 ASR 后端的置信度与稳定性联合精修门控，避免无条件调用 LLM；
2. 提出稳定前缀和动态活动后缀机制，降低流式文本跳动和重复计算；
3. 提出保护实体、结构化输出、自动验证和失败回退组成的可控精修链路；
4. 建立逐字参考与意图参考并存的流式评测方法，同时衡量质量、忠实度、延迟和计算成本。

其中第一至第三点构成方法贡献，第四点构成评测贡献。Whisper/Qwen3-ASR 支持、Web 前端和在线/离线切换属于实验平台能力，不单独作为论文创新点。

## 18. 专利“上下文记忆池”的选择性接入

项目只采用对方向 A 有直接增益且不会引入错误反馈循环的部分：

- 使用 SQLite 保存用户确认的标准实体、别名、领域和保护策略；
- 每个转录会话维护有容量和 TTL 限制的内存实体记忆；
- 数据库实体直接标记为 `VERIFIED`；
- 新实体必须有多次独立高置信观察才能晋级为 `ACCEPTED`；
- 缺少 ASR 置信度时，新实体不得自动晋级；
- Refiner 前执行占位保护，校验失败后回退原始 ASR 文本；
- 纠错结果、ASR confidence 和 memory trust 分开记录，互不覆盖。

暂不接入语义向量数据库、低语义相似度纠错、多候选在线生成以及
“纠错后自动提高置信度”。语义检索仅保留为长会话离线消融选项。
