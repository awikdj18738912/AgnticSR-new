# 流式 ASR 精修门控模块设计方案

## 1. 文档目的

本文档设计 AgenticASR 的选择性 Refiner 门控模块，用于解决流式 ASR 中每次文本更新都调用 Refiner 所造成的重复计算、GPU 排队和延迟升高问题。

门控模块的职责不是修改文本，而是判断当前文本是否值得调用 Refiner：

```text
KEEP：当前不需要精修
DEFER：当前信息不足，等待更多上下文
REFINE：满足条件，调用 Refiner
```

门控模块需要和现有的 K=3 滑动窗口配合使用：

```text
门控模块：决定“要不要调用 Refiner”
K=3 窗口：决定“调用时给 Refiner 看多少文本”
Validator：决定“精修结果能不能接受”
```

## 2. 当前问题

当前项目存在两条主要流式路径：

### 2.1 `system.live_asr`

该路径已经实现了 K=3 滑动窗口。`StreamingRefinementSession` 会维护最近三个稳定 chunk，并对这三个 chunk 组成的活动窗口进行精修。

但是，当前 `session.add()` 默认每次都会调用 Refiner，因此：

```text
每个新 chunk 都会触发一次 Refiner
```

它限制了输入窗口长度，但没有减少调用次数。

### 2.2 `system.web_app`

浏览器 Web 服务目前不是 K=3 滑动窗口路径：

- 在线模式会对变化后的累计 ASR 假设进行精修。
- 离线整段模式在结束后精修完整文本。
- 离线流式模式使用 latest-only，避免任务排队，但当前任务仍可能是累计假设。
- 结束时通常还会对最终完整文本执行一次精修。

因此，Web 路径也需要接入统一的门控和 K=3 窗口机制。

## 3. 总体架构

推荐架构如下：

```text
ASR partial / stable chunk
          ↓
HypothesisTracker
          ↓
FeatureExtractor
          ↓
RefinementGate
   ┌──────┼──────┐
   │      │      │
 KEEP   DEFER  REFINE
   │      │      │
 原文   等待   取最近 K=3 chunk
                ↓
             EntityProtector
                ↓
             Refiner
                ↓
             Validator
          ┌─────┴─────┐
          │           │
        接受        回退原文
```

## 4. 模块职责

建议新增以下文件：

```text
system/hypothesis_tracker.py
system/feature_extractor.py
system/gating.py
```

### 4.1 `hypothesis_tracker.py`

负责保存最近的 ASR 假设并计算稳定性：

- 当前文本和上一版本的最长公共前缀比例。
- 两次假设之间的修订比例。
- 文本连续不变的次数。
- 尾部文本保持不变的时间。
- 是否出现回滚或历史文本修改。
- 当前文本是否已经达到稳定边界。

### 4.2 `feature_extractor.py`

负责从 ASR 和文本中提取门控特征：

- ASR confidence。
- 假设稳定性。
- VAD 静音时长。
- 口头语。
- 重复。
- 自我修正。
- 标点缺失。
- 实体和数字风险。
- 是否已经精修过相同文本。
- Refiner 调用冷却时间是否结束。

### 4.3 `gating.py`

负责将特征转换为 KEEP、DEFER 或 REFINE，并给出风险分数和可解释原因。

## 5. 数据结构设计

建议先定义以下数据结构：

```python
from dataclasses import dataclass
from enum import Enum


class GateDecision(str, Enum):
    KEEP = "KEEP"
    DEFER = "DEFER"
    REFINE = "REFINE"


@dataclass(frozen=True)
class GateInput:
    text: str
    sequence_id: int
    is_final: bool
    asr_confidence: float | None = None
    audio_end_ms: float | None = None
    received_at_ms: float | None = None
    backend: str = "unknown"


@dataclass(frozen=True)
class GateFeatures:
    lcp_ratio: float
    revision_ratio: float
    unchanged_updates: int
    tail_age_ms: float
    has_filler: bool
    has_repetition: bool
    has_self_correction: bool
    missing_punctuation: bool
    entity_risk: float
    low_confidence: float | None
    same_as_last_refined: bool
    cooldown_active: bool
    stable: bool


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    risk_score: float
    reasons: tuple[str, ...]
    features: GateFeatures
```

## 6. HypothesisTracker 设计

### 6.1 状态

Tracker 至少需要维护：

```python
class HypothesisTracker:
    previous_text: str
    current_text: str
    last_refined_text: str
    last_sequence_id: int
    unchanged_updates: int
    last_change_ms: float
    last_refine_ms: float | None
    last_refined_sequence_id: int | None
```

### 6.2 更新逻辑

每次收到 ASR 假设时：

```text
收到新文本
    ↓
与上一次文本比较
    ↓
计算最长公共前缀比例
    ↓
计算修订比例
    ↓
更新 unchanged_updates
    ↓
更新 tail_age_ms
    ↓
判断 stable
```

伪代码：

```python
def update(self, hypothesis: GateInput) -> TrackerState:
    if hypothesis.text == self.current_text:
        self.unchanged_updates += 1
    else:
        self.unchanged_updates = 0
        self.last_change_ms = hypothesis.received_at_ms

    lcp_ratio = longest_common_prefix_ratio(
        self.current_text,
        hypothesis.text,
    )
    revision_ratio = normalized_revision_ratio(
        self.current_text,
        hypothesis.text,
    )

    stable = (
        self.unchanged_updates >= 2
        or tail_age_ms(hypothesis) >= 800
        or hypothesis.is_final
    )

    self.previous_text = self.current_text
    self.current_text = hypothesis.text
    return state
```

### 6.3 初始稳定规则

第一版可使用以下默认值：

```text
连续 2 次假设不变 → stable
尾部保持 800 ms → stable
收到 final 事件 → stable
```

这些只是启动参数，不能直接作为最终实验结论，需要在开发集上调节。

### 6.4 历史回滚

如果 ASR 修改了已经提交的稳定文本：

```text
原文：我准备明天去北京
新假设：我准备明天去北
```

不能直接覆盖已经提交的文本。应当：

1. 产生 rollback event。
2. 重新打开最近 K=3 个 chunk。
3. 重新等待稳定。
4. 必要时强制 REFINE。

## 7. FeatureExtractor 设计

### 7.1 口头语特征

第一版可以使用词表：

```text
嗯
呃
那个
就是
然后
这个
怎么说
```

可以计算：

```text
口头语数量 / 文本字符数
```

### 7.2 重复特征

检测以下模式：

```text
明天明天去北京
我们我们需要讨论
项目进度项目进度
```

建议同时检测：

- 连续重复字词。
- 重复 bigram。
- 重复短语。
- ASR 假设之间反复替换的尾部。

### 7.3 自我修正特征

自我修正词包括：

```text
不对
我是说
应该是
不是
改成
准确地说
换句话说
```

出现自我修正时，应该提高风险分数，并向左扩大活动窗口。

### 7.4 标点特征

如果文本较长但没有句末标点，可以提高精修风险：

```text
文本长度 > 20
且没有 。！？.!?
```

但是，标点缺失本身不一定意味着必须立即调用 Refiner。如果文本仍不稳定，应优先 DEFER。

### 7.5 实体风险

实体匹配器发现以下情况时，提高 `entity_risk`：

```text
疑似标准实体
疑似同音错误实体
数字、日期和模型标识符
领域候选分数较高
候选实体之间存在冲突
```

例如：

```text
鬼灵们 → 鬼灵门
```

即使没有口头语，也可以触发 REFINE 或实体验证。

### 7.6 ASR 置信度

如果 ASR 提供置信度：

```python
low_confidence = 1.0 - asr_confidence
```

如果没有 confidence：

- 保持 `None`。
- 不伪造数值。
- 使用文本稳定性、重复和实体特征补足判断。

## 8. 风险分数

第一版可以采用规则加权：

```text
risk_score =
    0.25 × low_confidence
  + 0.20 × revision_risk
  + 0.20 × disfluency_risk
  + 0.15 × repetition_risk
  + 0.10 × punctuation_risk
  + 0.10 × entity_risk
```

所有分项都限制在 `[0, 1]`。

如果 confidence 缺失，可以重新归一化其他已知特征的权重，而不是把缺失 confidence 当成 0。

## 9. 门控决策

### 9.1 基础规则

```python
def decide(features: GateFeatures, *, is_final: bool) -> GateResult:
    if features.same_as_last_refined:
        return keep("SAME_AS_LAST_REFINED")

    if not features.stable and not is_final:
        return defer("UNSTABLE_HYPOTHESIS")

    if features.cooldown_active and not is_final:
        return defer("REFINER_COOLDOWN")

    if is_final and features.risk_score >= 0.45:
        return refine("FINAL_RISK")

    if features.risk_score >= 0.65:
        return refine("RISK_THRESHOLD")

    return keep("LOW_RISK")
```

### 9.2 初始阈值

```yaml
stable_updates: 2
stable_tail_age_ms: 800
refine_threshold: 0.65
final_refine_threshold: 0.45
min_refine_interval_ms: 600
max_pending_queue: 1
```

### 9.3 KEEP

KEEP 表示当前不值得调用 Refiner，但不能丢弃文本：

- 保留原始 chunk。
- 更新稳定前缀。
- 更新当前显示文本。
- 记录门控决策。
- 后续如果发生自我修正，允许重新打开最近窗口。

### 9.4 DEFER

DEFER 表示信息不足：

- 保留在活动区。
- 不调用 Refiner。
- 等待下一次 ASR 假设。
- 不把可能被修改的文本永久提交。

适用于：

- 半个实体。
- 未完成句子。
- ASR 仍在快速变化。
- 当前文本过短。
- 等待自我修正后半句。

### 9.5 REFINE

REFINE 表示需要调用 Refiner：

1. 取最近 K=3 个稳定或活动 chunk。
2. 执行实体保护。
3. 调用 Refiner。
4. 执行 Validator。
5. 接受精修结果或回退原文。

## 10. 强制精修条件

以下情况可以绕过普通阈值，直接触发 REFINE：

```text
检测到自我修正
检测到明显重复
实体候选分数 >= 0.90
数字或日期发生变化
长静音结束
句末标点出现且窗口有未处理内容
活动窗口超过最大长度
收到 final 事件且存在待处理内容
```

对于高风险实体，可以使用更高阈值或要求用户确认。

## 11. 与 K=3 滑动窗口结合

K=3 不应该被删除。它应该作为“调用 Refiner 时的局部上下文窗口”。

假设稳定 chunks 为：

```text
C1：今天我们
C2：讨论项目
C3：然后嗯下周
C4：不对下周一开始测试
```

门控结果：

```text
C1：KEEP
C2：KEEP
C3：DEFER
C4：REFINE
```

触发 REFINE 时只取：

```text
C2 + C3 + C4
```

精修结果：

```text
讨论项目，然后下周一开始测试。
```

最终拼接：

```text
今天我们 + 讨论项目，然后下周一开始测试。
```

### 11.1 `StreamingRefinementSession` 改造

当前接口类似：

```python
session.add(chunk)
```

建议改为：

```python
session.add(chunk, decision=GateDecision.KEEP)
session.add(chunk, decision=GateDecision.DEFER)
session.add(chunk, decision=GateDecision.REFINE)
```

伪代码：

```python
def add(self, chunk, decision):
    self.raw_chunks.append(chunk)

    if decision == GateDecision.DEFER:
        return self.current_raw_transcript()

    if decision == GateDecision.KEEP:
        self.commit_stable_raw_chunks()
        return self.current_transcript()

    if decision == GateDecision.REFINE:
        raw_window = self.raw_chunks[-self.window_size:]
        refined = self.refiner.refine(join_chunks(raw_window))
        return self.replace_active_window(refined)
```

### 11.2 已提交前缀和活动窗口

系统需要区分：

```text
committed_prefix：已经稳定提交，不再被普通精修修改
readonly_context：给 Refiner 参考，但不可编辑
active_span：允许 Refiner 修改的最近窗口
```

K=3 主要对应活动窗口及其上下文，不代表整个会话永远只能保留三个 chunk。

## 12. 调用节流和 latest-only

门控还需要防止多个 Refiner 任务排队：

```text
同一文本不重复精修
两次精修至少间隔 600 ms
同一会话只允许一个活动 Refiner 任务
只保留最新 pending 请求
旧 sequence_id 的结果直接丢弃
```

例如：

```text
Refiner 正在处理 A
新文本 B 到达
新文本 C 到达
```

不应该排队成：

```text
A → B → C
```

而应该：

```text
处理 A
丢弃 B
A 完成后只处理 C
```

## 13. 与 Validator 的关系

门控只负责“是否调用”，不负责判断模型结果是否正确。

REFINE 后必须执行 Validator：

```text
实体是否被改变
数字和日期是否被改变
是否出现重复生成
是否出现截断
是否新增原文不存在的事实
编辑是否超出活动窗口
是否满足输出格式
```

如果 Validator 失败：

```text
使用原始 ASR 文本
记录 fallback_raw
记录 reject_reason
```

## 14. Web 服务接入

当前 `system/web_app.py` 需要从直接调用：

```python
refine_update(raw_text, ...)
```

改为：

```python
hypothesis = tracker.update(gate_input)
features = feature_extractor.extract(hypothesis)
decision = gate.decide(features)

if decision.decision == GateDecision.REFINE:
    queue_refinement(last_k_chunks(3))
elif decision.decision == GateDecision.DEFER:
    send_status("等待更多上下文")
else:
    send_raw_or_previous_result()
```

在线模式、离线流式模式和 `system.live_asr` 应尽量共用同一个 Gate，避免三套入口产生不同的门控行为。

## 15. 日志协议

每次门控都应写入 JSONL：

```json
{
  "session_id": "session-001",
  "sequence_id": 12,
  "raw_text": "我准备明天明天去北京",
  "gate": {
    "decision": "REFINE",
    "risk_score": 0.78,
    "reasons": [
      "REPETITION",
      "STABLE_BOUNDARY"
    ],
    "features": {
      "lcp_ratio": 0.94,
      "revision_ratio": 0.06,
      "unchanged_updates": 2,
      "tail_age_ms": 920,
      "has_filler": false,
      "has_repetition": true,
      "has_self_correction": false,
      "missing_punctuation": true,
      "entity_risk": 0.0,
      "cooldown_active": false,
      "stable": true
    }
  },
  "window": {
    "size": 3,
    "chunk_start": 9,
    "chunk_end": 12
  }
}
```

这些日志可用于后续训练学习型 Gate。

## 16. 测试计划

### 16.1 稳定性测试

```text
正常文本连续不变 2 次 → stable
文本快速变化 → unstable
超过 800 ms 没有变化 → stable
收到 final → stable
```

### 16.2 决策测试

```text
正常稳定句子 → KEEP
未完成短句 → DEFER
ASR 快速变化 → DEFER
口头语 → REFINE
重复 → REFINE
自我修正 → REFINE
疑似实体错误 → REFINE
同一文本重复到达 → KEEP
final + 未处理内容 → REFINE
final + 无风险且已精修 → KEEP
```

### 16.3 K=3 测试

需要验证：

1. REFINE 时只处理最近三个 chunk。
2. KEEP 不会触发模型调用。
3. DEFER 不会提前提交未稳定文本。
4. 已提交前缀不会被普通窗口修改。
5. 自我修正能够重新打开最近窗口。
6. final 事件能够处理最后未精修内容。

### 16.4 资源指标

至少记录：

```text
Refiner 调用次数
调用减少比例
KEEP/DEFER/REFINE 分布
精修接受率
精修回退率
p50/p95 延迟
平均输入字符数
GPU 推理时间
```

## 17. 推荐实现顺序

### 阶段一：规则门控

新增：

```text
system/hypothesis_tracker.py
system/feature_extractor.py
system/gating.py
```

先不训练模型，只验证规则决策和日志是否正确。

### 阶段二：接入 K=3

让 `StreamingRefinementSession` 支持 KEEP、DEFER、REFINE，并确保只有 REFINE 才会调用模型。

### 阶段三：接入 Validator

对精修结果执行实体、数字、重复、截断和编辑范围检查，失败时回退原文。

### 阶段四：接入 Web 服务

让 Web 在线、离线流式和本地 `live_asr` 统一使用 Gate、K=3 和 latest-only 调度。

### 阶段五：训练学习型 Gate

积累日志后，把每次决策标注为：

```text
精修有收益
精修无收益
精修改错
不应该精修
```

再训练 Logistic Regression、LightGBM 或小型 MLP，并与规则 Gate 对比。

## 18. 初始完成标准

第一版门控模块完成后，应满足：

1. 系统支持 KEEP、DEFER、REFINE 三种决策。
2. 每次决策都有风险分数和可解释原因。
3. 正常稳定文本不会无条件调用 Refiner。
4. 不稳定文本能够等待更多上下文。
5. 重复、口头语、自我修正和实体风险能够触发精修。
6. REFINE 时仍使用 K=3 滑动窗口。
7. K=3 窗口只在门控允许时调用 Refiner。
8. 同一文本不会重复调用 Refiner。
9. final 事件能够处理剩余活动文本。
10. 精修失败时能够回退原始 ASR 文本。
11. 日志能够统计调用次数、延迟和门控分布。

## 19. 最终原则

门控不是简单地“每隔几次调用一次 Refiner”，而是根据文本稳定性、错误风险、实体风险和上下文完整性做决策。

最终推荐架构为：

```text
规则 Gate
    ↓
K=3 滑动窗口
    ↓
实体保护
    ↓
Refiner
    ↓
Validator
    ↓
接受精修或回退原文
```

K=3 不需要删除。它负责控制精修上下文范围；门控负责减少不必要的精修调用；Validator 负责防止错误精修进入最终结果。
