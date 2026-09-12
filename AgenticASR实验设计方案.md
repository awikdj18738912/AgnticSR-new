# AgenticASR 实验设计与评测方案

## 1. 实验目标

本方案用于评估当前项目的三项核心能力：

1. Refiner 是否在保留原意的前提下改善 ASR 文本。
2. 实体检测和替换是否能纠正错写实体，同时避免误替换正确文本。
3. K=3 滑动窗口及缓存复用是否能降低离线/流式结束阶段的延迟，同时保持最终质量。

实验应分别回答以下问题：

- **RQ1：** Refiner 相比原始 ASR，能否降低 CER、WER、MER，并改善 Content、Format、Filter、Rephrase 四类质量？
- **RQ2：** 精确别名、拼音候选、提示模式和自动替换分别带来多少实体召回收益与误替换风险？
- **RQ3：** K=1、K=3、K=5 的质量、延迟和稳定性有什么差异？
- **RQ4：** 结束时复用窗口缓存，能否减少 Refiner 调用和超时，而不降低最终文本质量？
- **RQ5：** 内容丢失、占位符、重复生成等安全规则能拦截多少真实错误，又会误拦截多少正常精修？
- **RQ6：** 可解耦的精修前门控能减少多少 Refiner 调用和延迟，同时是否保持精修质量？

## 2. 实验原则

所有系统版本必须使用相同的音频、原始 ASR 输出、实体库和 Refiner 模型。阈值只能在开发集上调整，测试集只运行一次最终配置。

质量实验优先冻结 ASR 输出，再比较不同后处理策略。这样可以排除 ASR 每次推理差异，确保观察到的是实体模块或 Refiner 的变化。

延迟实验必须从真实音频重新运行，因为冻结文本无法测量音频上传、流式更新、窗口缓存和结束阶段延迟。

建议固定以下条件：

- Refiner 使用同一 checkpoint。
- `do_sample=false`，保证输出可复现。
- 使用相同 GPU、CUDA、PyTorch 和 Transformers 版本。
- 每个延迟配置先预热 3 次，再正式运行至少 5 次。
- 流式实验使用相同的音频块大小和发送节奏。
- 每次实验保存 Git commit、配置文件、命令、硬件信息和原始逐样本结果。

## 3. 当前系统配置快照

当前实体配置位于 `system/configs/entity_matching.json`：

| 配置项 | 当前值 |
|---|---:|
| 拼音权重 | 0.55 |
| 字符权重 | 0.25 |
| 长度权重 | 0.10 |
| 领域权重 | 0.08 |
| 优先级权重 | 0.02 |
| 候选提示阈值 | 0.80 |
| 自动替换阈值 | 0.80 |
| 自动替换最低拼音分 | 0.88 |
| 最佳与次佳候选最低分差 | 0.12 |
| 自动替换类型 | TERM、PROJECT、MODEL |
| 高风险类型 | PERSON、ORG |

当前窗口与安全配置：

| 配置项 | 当前值 |
|---|---:|
| 单个文本块最大长度 | 80 字符 |
| 滑动窗口大小 | K=3 |
| 最终精修最长等待 | 300 秒 |
| 严重内容丢失长度比例 | 0.65 |
| 实体模糊模式 | auto |
| 精修前门控 | 默认 off；实验组 conservative |

实验开始前应复制这些配置，而不是直接覆盖当前生产配置。

## 4. 数据集设计

### 4.1 通用精修数据

优先使用仓库已经支持的 AASR-Bench。它包含 917 条样本和 Content、Format、Filter、Rephrase 四类原子评分项，可用于评估整体精修质量。

每条数据至少需要：

- `source_record_id`
- 音频路径
- 原始口语文本
- 人工清洗参考文本
- ASR 原始转写
- 场景和语言

### 4.2 实体纠错数据

需要单独建立实体标注集。建议最低 600 条，正式论文实验建议 1000 条以上。

| 子集 | 最低数量 | 目的 |
|---|---:|---|
| 实体错写正例 | 200 | 测量能否正确纠错 |
| 实体正确负例 | 200 | 测量是否误替换 |
| 同音/近音困难样本 | 100 | 测量“李飞鱼/厉飞羽”类问题 |
| 中英混合、数字、缩写 | 50 | 测量模型名和项目名 |
| 无实体普通文本 | 50 | 测量候选模块是否过度触发 |

正例、负例必须成对设计。例如：

- 正例：上下文明确指向小说角色，ASR 输出“李飞鱼”，标准实体为“厉飞羽”。
- 困难负例：上下文确实在介绍姓名为“李飞鱼”的人物，系统应保留原文。
- 候选冲突：词库同时存在多个同音或近音实体，检查候选间隔是否能阻止误替换。

建议的 JSONL 标注格式：

```json
{
  "sample_id": "entity-0001",
  "audio": "data/audio/entity-0001.wav",
  "language": "Chinese",
  "domain": "novel",
  "raw_text": "韩立想起了李飞鱼。",
  "reference_text": "韩立想起了厉飞羽。",
  "entities": [
    {
      "observed": "李飞鱼",
      "canonical": "厉飞羽",
      "entity_type": "PERSON",
      "should_replace": true
    }
  ]
}
```

实体边界应以原始转写为准。若一个样本含多个实体，每个实体分别标注。

### 4.3 长音频与流式数据

建议准备至少 30 条长音频，覆盖以下时长：

- 5～10 分钟：10 条
- 10～30 分钟：10 条
- 30～60 分钟：10 条

内容至少覆盖中文、英文、中英混合、实体密集对话和普通访谈。当前曾出现的 58,840 字符长转录应作为固定压力测试样本。

### 4.4 数据划分

自建数据按照 60%/20%/20% 划分为训练或规则开发集、验证集和测试集。必须按录音来源、说话人和原始视频分组划分，禁止把同一录音切片分到不同集合。

实体实验建议额外做两种划分：

- **Seen-entity：** 测试实体已在开发集出现，但上下文和错写形式不同。
- **Unseen-alias：** 标准实体在词库中，但测试错写形式没有在开发集出现。

## 5. 对照系统

### 5.1 主实验版本

| 编号 | 系统 | 用途 |
|---|---|---|
| S0 | 原始 ASR | 最低基线 |
| S1 | ASR + Refiner，不加载实体库 | 测量纯精修收益 |
| S2 | S1 + 精确实体/别名 | 测量确定性实体规则收益 |
| S3 | S2 + fuzzy shadow | 只记录候选，不修改文本 |
| S4 | S2 + fuzzy hint | 候选提供给 Refiner，不做确定性自动替换 |
| S5 | S2 + fuzzy auto | 当前完整实体策略 |
| S6 | S5 + 上下文候选重排 | 后续上下文实体方案 |

S6 尚未实现时，主表先报告 S0～S5，并将 S6 放入后续实验。

### 5.2 滑动窗口与结束策略

| 编号 | 窗口 | 结束策略 |
|---|---:|---|
| W0 | 无窗口 | 结束后重新精修全文 |
| W1 | K=1 | 复用缓存，只处理活动窗口 |
| W3 | K=3 | 当前方案 |
| W5 | K=5 | 更长上下文对照 |
| W3-Full | K=3 | 实时用窗口，结束后仍重新精修全文 |

W3 与 W3-Full 的比较直接回答缓存复用是否有效。不要将所有实体版本与所有窗口版本做完整笛卡尔积；先分别选优，再组合最终版本，避免实验数量失控。

### 5.3 安全校验消融

| 编号 | 启用规则 |
|---|---|
| G0 | 不做输出校验 |
| G1 | 仅长度与空输出校验 |
| G2 | G1 + 重复生成检测 |
| G3 | G2 + 占位符数量、顺序与格式校验 |
| G4 | G3 + 实体句界校验与严格重试，当前完整方案 |

### 5.4 精修前门控消融

门控模块只负责决定是否调用 Refiner，不参与实体确定性替换，也不替代输出安全校验。使用完全相同的冻结 ASR 输入比较：

| 组别 | `--refinement-gate-mode` | 含义 |
|---|---|---|
| R0 | `off` | 原始基线，每个非空片段都调用 Refiner |
| R1 | `conservative` | 跳过极短片段，以及高置信、句子完整且无清理信号的片段 |

R1 在 ASR 未提供置信度时继续调用 Refiner；有实体提示时也强制进入 Refiner。逐样本记录 `refiner_executed`、`refinement_gate_decisions` 和 `refinement_gate_skipped_segments`，重点比较模型调用/千字、延迟 p50/p95、CER/MER、AASR 和门控误跳过率。

## 6. 指标定义

### 6.1 通用文本指标

仓库已有 `experiments/scripts/text_metrics.py`，可以计算：

- 中文字符错误率 CER
- 英文词错误率 WER
- 中英混合错误率 MER
- AASR-Bench 总分
- Content、Format、Filter、Rephrase 分类得分

CER/WER/MER 越低越好，AASR-Bench 得分越高越好。精修任务不能只报告错误率，因为合理的去口癖和重写可能在字面上与参考答案不同。

### 6.2 实体指标

以“是否应替换”和“替换后的 canonical 是否正确”为判断依据：

```text
TP：需要替换，并替换为正确 canonical
FP：不应替换却发生替换，或替换成错误 canonical
FN：需要替换但没有替换，或替换结果错误
TN：不应替换并正确保留
```

报告：

- Entity Correction Precision = TP / (TP + FP)
- Entity Correction Recall = TP / (TP + FN)
- Entity Correction F1
- False Replacement Rate = FP / (FP + TN)
- Exact、Fuzzy、Hint 三种来源分别统计
- 按 PERSON、ORG、TERM、PROJECT、MODEL 分组统计
- 按领域、实体长度和拼音相似度分桶统计
- top-1/top-3 候选召回率
- 候选分差分布及可靠性曲线

自动替换首先追求 Precision。高风险实体即使 Recall 较低，也不能用明显增加误替换来换取。

### 6.3 精修安全指标

- 严重内容丢失率
- 新增事实或实体率
- 占位符损坏率
- 重复生成率
- 异常片段回退率
- 严格重试成功率
- 正常精修误拦截率
- 最终文本中未闭合引号、括号比例
- 原始实体保留率和规范实体恢复率

安全规则需要同时报告“拦住多少坏结果”和“误伤多少好结果”。仅报告回退次数没有意义。

### 6.4 流式和性能指标

- 首次原始转写延迟
- 首次精修结果延迟
- 每次窗口更新延迟 p50/p95/p99
- 点击结束到最终结果的延迟 p50/p95/p99
- Real-Time Factor（总处理时间 / 音频时长）
- 每分钟 Refiner 调用次数
- 每千字符 Refiner 调用次数
- 缓存复用率
- 最终阶段新增模型调用次数
- 超时率和 `refiner_busy` 比例
- GPU 峰值显存、平均利用率和吞吐量
- 实时最终文本与结束后最终文本的一致率

缓存复用率建议定义为：

```text
复用的已提交窗口数 / 最终文本对应的全部窗口数
```

## 7. 实验执行顺序

### 阶段 A：建立冻结输入

1. 对所有音频只运行一次 ASR，保存原始输出。
2. 人工检查 `source_record_id`、语言和空输出。
3. 保存原始输出文件的 SHA-256。
4. 后续所有质量实验读取同一份原始 JSONL。

输出建议放在：

```text
results/experiments/datasets/frozen_asr.jsonl
results/experiments/datasets/entity_test.jsonl
results/experiments/datasets/long_audio_manifest.jsonl
```

### 阶段 B：实体候选与阈值实验

先运行 shadow 模式收集候选，不修改文本。使用开发集搜索以下参数：

- `auto_score`：0.80、0.84、0.88、0.92
- `auto_min_pinyin`：0.84、0.88、0.92、0.96
- `auto_min_margin`：0.08、0.12、0.16、0.20

第一轮固定现有权重，只搜索阈值。选出合理阈值后，再做少量权重消融：

- 拼音为主：0.65/0.20/0.07/0.06/0.02
- 当前配置：0.55/0.25/0.10/0.08/0.02
- 字符增强：0.45/0.35/0.10/0.08/0.02

阈值选择目标：先满足自动替换 Precision 和误替换率要求，再在满足约束的配置中选择 Recall 最高者。

### 阶段 C：窗口和缓存实验

在同一组长音频上运行 W0、W1、W3、W5、W3-Full。每个版本至少重复 5 次延迟测量，质量只需对确定性输出评测一次。

重点比较：

- W3 与 W3-Full 的最终文本质量差异。
- W3 与 W3-Full 的最终等待时间、模型调用次数和超时率。
- K=1、3、5 对跨句自我修正和实体上下文的影响。
- 最终 ASR 修改早期文本时，缓存失效与重新精修是否正确。

### 阶段 D：安全校验消融

将预先收集的正常输出与异常输出固定下来，离线运行 G0～G4。异常集至少包含：

- 长文本被压缩成一句话
- 模型循环重复
- 实体占位符丢失、重复、乱序或格式变化
- 两个实体句子被合并成列表
- 正常去除大量口癖和重复致谢
- 删除“不是、但是、只不过、还是”等关键逻辑内容
- 引号和括号破坏

此阶段不重新调用模型，只评测 validator，保证每个规则看到完全相同的候选输出。

### 阶段 E：最终端到端实验

组合开发集上选出的实体配置、K 值、缓存策略和安全规则，在保留测试集上运行一次。最终主表至少包含 S0、S1、S2、S5 和最终系统。

## 8. 当前仓库可直接运行的命令

### 8.1 运行现有测试

```bash
conda run --no-capture-output -n agentic-asr \
  python -m unittest discover -s tests
```

### 8.2 对冻结 ASR 输出运行 Refiner

```bash
conda run --no-capture-output -n agentic-asr \
  python experiments/scripts/postprocess_asr.py \
  results/experiments/datasets/frozen_asr.jsonl \
  results/experiments/runs/s5_auto/output.jsonl \
  --model /home/aim0/data/models/ASR/AgenticASR-Refiner \
  --entity-db data/entities.db \
  --entity-domain general \
  --entity-fuzzy-mode auto \
  --refinement-gate-mode off \
  --batch-size 8 \
  --overwrite
```

将 `--entity-fuzzy-mode` 分别设为 `shadow`、`hint` 和 `auto`，生成对应结果。S1 不传 `--entity-db`。

门控 A/B 对比时保持其他参数不变，R0 使用 `--refinement-gate-mode off`，R1 只改为 `--refinement-gate-mode conservative`。不要同时调整实体阈值或窗口大小。

### 8.3 运行 AASR-Bench Judge

```bash
conda run --no-capture-output -n agentic-asr \
  python experiments/scripts/main.py \
  results/experiments/runs/s5_auto/output.jsonl \
  --rubric /path/to/rubric.json \
  --results results/experiments/runs/s5_auto/judge.jsonl \
  --summary results/experiments/runs/s5_auto/summary.json \
  --api-url http://127.0.0.1:8000/v1 \
  --model /path/to/judge-model
```

## 9. 建议补充的实验脚本

以下脚本当前仓库还没有，应在正式跑实验前实现：

```text
experiments/scripts/evaluate_entity_correction.py
experiments/scripts/replay_streaming_session.py
experiments/scripts/summarize_streaming_runtime.py
experiments/scripts/compare_runs.py
```

职责分别为：

- `evaluate_entity_correction.py`：计算实体 Precision、Recall、F1、误替换率和分桶指标。
- `replay_streaming_session.py`：用固定音频块节奏重复运行 offline/streaming 模式。
- `summarize_streaming_runtime.py`：统计窗口调用、缓存命中、最终延迟、超时和显存。
- `compare_runs.py`：按相同 `sample_id` 做配对比较并输出置信区间。

这些脚本的输出必须保留逐样本结果，不能只保存汇总数字。

## 10. 统计检验

质量指标采用配对比较，因为不同系统处理的是同一批样本：

- CER、WER、MER 和 AASR-Bench 得分：使用 10,000 次 paired bootstrap，报告 95% 置信区间。
- 实体“替换正确/错误”决策：使用 McNemar 检验。
- 延迟：报告中位数、p95、p99，并使用 bootstrap 比较中位数差异。
- 同时报告绝对值和相对变化，不能只报告百分比提升。

若测试多个阈值，显著性检验只能用于最终选择后的测试集结果，不能把测试集反复用于调参。

## 11. 人工评审

自动指标之外，建议随机抽取至少 200 条样本进行盲评。评审者不能知道文本来自哪个系统。

每条文本按以下维度评分：

- 原意是否完整保留
- 是否存在事实新增或语义删除
- 口癖和重复是否处理合理
- 标点和书面表达是否自然
- 实体是否正确

至少由两名评审者独立标注，报告一致率或 Cohen's kappa。意见不一致的样本由第三人裁决。

## 12. 推荐验收标准

以下数值适合作为第一版发布门槛，可在开发集规模扩大后调整：

| 指标 | 建议门槛 |
|---|---:|
| 低风险实体自动替换 Precision | ≥ 98% |
| 实体困难负例误替换率 | ≤ 1% |
| 实体 Recall 相对精确匹配 | 提升至少 10 个百分点 |
| 占位符损坏进入最终文本 | 0 |
| 严重内容丢失进入最终文本 | ≤ 0.5% 样本 |
| 长音频最终精修超时率 | 0 |
| 缓存复用后 Refiner 调用减少 | ≥ 80% |
| 已缓存长音频最终等待 p95 | ≤ 10 秒 |
| W3 相对 W3-Full 的 CER/MER 绝对差 | ≤ 0.2 个百分点 |
| W3 相对 W3-Full 的 AASR 总分差 | ≤ 0.01 |

PERSON、ORG 当前属于高风险类型，不应使用低风险类型的自动替换门槛。若后续允许自动替换，必须单独报告，推荐 Precision 门槛不低于 99%。

## 13. 结果表模板

### 13.1 主结果

| 系统 | CER ↓ | WER ↓ | MER ↓ | AASR ↑ | Content ↑ | Format ↑ | Filter ↑ | Rephrase ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 Raw ASR |  |  |  |  |  |  |  |  |
| S1 Refiner |  |  |  |  |  |  |  |  |
| S2 Exact |  |  |  |  |  |  |  |  |
| S5 Fuzzy Auto |  |  |  |  |  |  |  |  |
| Final |  |  |  |  |  |  |  |  |

### 13.2 实体结果

| 系统 | Precision ↑ | Recall ↑ | F1 ↑ | 误替换率 ↓ | top-3 召回 ↑ |
|---|---:|---:|---:|---:|---:|
| Exact |  |  |  |  |  |
| Shadow 候选上限 |  |  |  |  |  |
| Hint |  |  |  |  |  |
| Auto |  |  |  |  |  |
| Context Rerank |  |  |  |  |  |

### 13.3 窗口和性能结果

| 系统 | 最终延迟 p50 | p95 | 超时率 | 模型调用/千字 | 缓存复用率 | GPU 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| W0 |  |  |  |  |  |  |
| W1 |  |  |  |  |  |  |
| W3 |  |  |  |  |  |  |
| W5 |  |  |  |  |  |  |
| W3-Full |  |  |  |  |  |  |

## 14. 实验产物目录

每个实验建立独立目录：

```text
results/experiments/
├── datasets/
├── runs/
│   └── 2026-09-11_s5_auto_w3/
│       ├── manifest.json
│       ├── config/
│       ├── command.txt
│       ├── output.jsonl
│       ├── runtime.jsonl
│       ├── judge.jsonl
│       ├── summary.json
│       └── errors.jsonl
└── tables/
```

`manifest.json` 至少记录：

```json
{
  "run_id": "2026-09-11_s5_auto_w3",
  "git_commit": "填写实际 commit",
  "asr_model": "Qwen3-ASR-1.7B",
  "refiner_model": "AgenticASR-Refiner",
  "entity_config_version": 1,
  "window_size": 3,
  "random_seed": 42,
  "device": "填写 GPU 型号",
  "status": "complete"
}
```

## 15. 推荐的最小可行实验

如果前期资源有限，先完成以下四组即可形成可信结论：

1. 用 AASR-Bench 比较 S0、S1、S2、S5。
2. 用 600 条实体集比较 Exact、Hint、Auto，并搜索三个实体阈值。
3. 用 30 条长音频比较 W3 与 W3-Full。
4. 用真实异常片段评测 G1 与 G4 的坏结果拦截率和正常结果误拦截率。

完成这四组后，再决定是否投入上下文候选重排、音频特征实体检索或 Refiner 再训练。

## 16. 最终检查清单

- [ ] ASR 原始输出已经冻结并校验哈希。
- [ ] 开发集和测试集按录音来源隔离。
- [ ] 每个系统使用相同模型和输入。
- [ ] 测试集没有参与阈值选择。
- [ ] 保存逐样本输出和失败原因。
- [ ] 分别报告质量、实体、安全和延迟指标。
- [ ] 长音频记录模型调用数和缓存复用率。
- [ ] 结果包含置信区间或显著性检验。
- [ ] 人工评审采用盲评。
- [ ] 配置、Git commit、环境和命令可以复现。
