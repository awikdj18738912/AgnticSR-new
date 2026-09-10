# AgenticASR 当前工作总结与后续工作安排

> 文档版本：2026-09-09
> 统计范围：当前工作区代码、文档、测试和运行产物
> 项目状态：研究型可运行原型，已完成一轮“术语保护与 Web 交互增强”，方向 A 的完整研究闭环尚未完成。

## 1. 总体结论

AgenticASR 当前已经具备从 ASR 文本到精修文本的基础能力，并同时覆盖批处理、离线文件、在线麦克风和离线流式文件等使用方式。最近一轮工作重点放在“实体不要被精修模型改错”以及“浏览器端能够管理和观察术语”的工程能力上，已经形成了 SQLite 术语库、实体保护模块、会话级实体记忆、流式 Refiner 调度和 Web 管理界面。

当前最重要的判断是：

- 已完成：基础 ASR/Refiner 链路、批处理与流式入口、术语库 CRUD、确定性保护规则、会话记忆策略、Web UI 交互增强、基础单元测试。
- 部分完成：精修结果质量拦截、实体审计、异步流式精修、结果日志。已有可用实现，但还不是方向 A 设计中的完整 Validator、动态窗口和门控系统。
- 尚未完成：统一 ASR Hypothesis 协议、KEEP/DEFER/REFINE 置信度门控、稳定前缀与 active span、结构化 Refiner 输出、语义校验、双参考数据集、学习型门控和完整实验矩阵。

因此，当前版本适合作为“功能原型和安全保护模块的开发基线”，还不适合作为方向 A 的最终实验版本或论文最终结果版本。

## 2. 当前仓库结构与主要入口

| 目录或文件 | 当前职责 |
|---|---|
| `pipeline/` | 生成 Oral/Clean 配对数据、模拟 ASR、质量控制、去重和 SFT 导出 |
| `experiments/scripts/` | 批量精修、ASR + Refiner 联合推理、AASR-Bench 评测相关脚本 |
| `system/` | 在线 ASR、VAD、浏览器 WebSocket 服务、Refiner 和会话运行逻辑 |
| `system/web/` | 浏览器端录音、文件上传、模式切换、结果展示和术语库管理 |
| `tests/` | 实体保护、会话记忆和精修结果安全检查的单元测试 |
| `DIRECTION_A_IMPLEMENTATION_PLAN.md` | 方向 A 的总体架构、阶段计划、评测指标和验收标准 |
| `results/` | 当前已有的端到端和 Web 会话 JSON/JSONL 运行结果 |
| `data/entities.db` | 当前工作区中的 SQLite 术语库实例 |

主要运行入口如下：

```bash
# 批量 ASR 文本精修
python experiments/scripts/postprocess_asr.py \
  /path/to/asr_output.jsonl \
  /path/to/refined_output.jsonl \
  --model /path/to/refiner-checkpoint

# 浏览器 Web 服务
python -m system.web_app \
  --refiner-model /path/to/AgenticASR-Refiner \
  --asr-url http://127.0.0.1:8766 \
  --entity-db data/entities.db \
  --output results/web/session.jsonl
```

## 3. 已完成的基础工作

### 3.1 数据生成、训练和评测基础

项目原有的研究代码已经形成一条较完整的数据和模型流程：

1. `pipeline/scripts/01_seed_and_oral.py` 到 `05_finalize.py` 生成口语文本、清洁文本、模拟 ASR 假设，并进行质量控制和去重。
2. `pipeline/scripts/export_sft.py` 将最终数据导出为 LLaMA Factory 使用的 SFT 格式。
3. `experiments/scripts/postprocess_asr.py` 对包含 `source_record_id` 和 `output.raw_text` 的 JSONL ASR 结果执行批量精修。
4. `experiments/scripts/transcribe_and_refine.py` 串联 Whisper 或 Qwen3-ASR 与 Refiner，提供离线端到端推理入口。
5. AASR-Bench 评测脚本和 JSONL 结果格式已经存在，可作为后续对比实验基础。

### 3.2 流式系统和前端基础

当前 `system/` 已具备以下能力：

- VAD、在线 ASR 和 Refiner 的基础运行链路。
- Qwen3-ASR 和 Whisper 两类 ASR 后端的浏览器服务接入方式。
- `/stream/start`、`/stream/chunk`、`/stream/finish` HTTP ASR 协议，以及 `/ws/stream` 浏览器 WebSocket 协议。
- 麦克风在线模式、文件离线整段模式、文件离线流式模式三种前端工作方式。
- Refiner 模型设备、最大生成长度、语言、输出 JSONL 等命令行配置。
- WebSocket 发送锁、连接关闭处理和后台流式精修任务，降低浏览器断开时的异常概率。

这些能力构成当前项目的可运行基线，但还没有完全升级为方向 A 规划中的“统一假设协议 + 稳定性分析 + 置信度门控 + 动态活动窗口”架构。

## 4. 最近一轮已经完成的工作

### 4.1 方向 A 实施方案文档

新增 `DIRECTION_A_IMPLEMENTATION_PLAN.md`，明确了统一 ASR Hypothesis、稳定前缀、活动后缀、动态窗口、KEEP/DEFER/REFINE、实体保护、Validator、双参考数据集、流式指标、消融实验和阶段验收标准。

该文档是后续开发的设计依据，但其中列出的模块不能全部视为已经实现。

### 4.2 SQLite 术语库和命令行管理

新增 `system/entity_store.py` 和 `system/manage_entities.py`：

- 保存标准名称、别名、实体类型、领域、规范化策略、优先级和启用状态。
- 支持 `preserve` 和 `normalize` 两种策略。
- 支持通用领域和指定领域；选择某个领域时同时加载 `general` 实体。
- 支持新增、更新、启用、停用、删除、别名更新和列表查询。
- SQLite 负责持久化，推理时加载到内存，避免每个 ASR partial 都查询数据库。
- 术语库实体默认视为用户验证过的 `VERIFIED` 实体。

例如：

```bash
python -m system.manage_entities --db data/entities.db add AgenticASR \
  --alias "Agentic SR" \
  --type PROJECT \
  --policy normalize \
  --priority 10

python -m system.manage_entities --db data/entities.db list
```

### 4.3 确定性实体保护模块

新增 `system/protection.py`，目前可以识别和保护 URL、邮箱、日期、时间、整数、小数、百分比、带连接符或版本号形式的英文标识符、英文缩写，以及 SQLite 术语库中精确命中的标准名称和别名。

保护流程会把命中的片段替换为 `⟦P000⟧` 形式的占位符，并记录原文、实体类型、来源和恢复文本。恢复阶段会检查占位符是否缺失、重复、未知、重排或格式错误；失败时返回原始文本和拒绝原因。配置为 `normalize` 的显式别名还可以在精确匹配后恢复为标准名称，并记录每一次替换。

### 4.4 会话级实体记忆

新增 `system/session_memory.py`，实现了有限容量、TTL 和信任等级的会话记忆：

- 数据库实体可以立即进入 `VERIFIED`。
- 新观察首先处于 `PROVISIONAL` 或 `OBSERVED`。
- 只有来自不同稳定段、且至少两次高置信度观察，才可以晋级为 `ACCEPTED`。
- 缺少 ASR confidence 时，不会自动晋级为可信实体。
- 重复的同一个 partial 假设不会被计为独立证据。
- 记忆具有容量上限和过期时间，避免错误实体无限积累。

这部分实现了“纠错结果不能直接提高实体可信度”的安全原则。

### 4.5 Web 端实体管理和三种处理模式

更新 `system/web/index.html` 和 `system/web_app.py` 后，浏览器端新增：

- 术语领域选择器和术语库管理面板。
- 实体搜索、增加、编辑、启用/停用和删除。
- 实体类型、领域、别名和规范化策略设置。
- 在线麦克风、离线整段文件、离线流式文件三种处理模式。
- 精修历史面板，展示原文、精修文本、实体规范化、命中术语、拒绝原因和审计提示。
- 运行状态、ASR 置信度、Refiner 延迟等信息展示。

服务端新增 `/api/entities` 相关接口，并将实体领域、保护实体、规范化记录、精修接受状态和拒绝原因写入 Web 会话 JSONL。

### 4.6 流式精修调度和安全输出

当前离线流式模式已经支持：

- ASR 文本更新先发送给前端，避免 Refiner 延迟阻塞转录显示。
- 只保留最新待精修文本，避免后台任务无限排队。
- 精修任务在后台线程执行。
- 会话结束前等待最后一项待处理精修。
- 连接断开时停止后台任务并尝试结束 ASR 会话。

新增 `system/refinement_guard.py`，提供以句末标点和软分隔符为优先级的分段、默认约 200 字符的精修窗口、中文文本合并，以及空输出、重复句、重复短语和明显截断输出的拒绝判断。

## 5. 当前验证结果

已执行以下检查：

```bash
python -m unittest discover -s tests -v
python -m compileall -q system experiments/scripts tests
git diff --check
```

结果：

- `tests/` 当前共 20 项单元测试，全部通过。
- Python 编译/语法检查通过。
- Git 差异空白检查通过。
- 已覆盖实体库 CRUD、领域过滤、数字和标识符保护、占位符缺失/未知/重排、别名规范化、会话记忆晋级和精修重复/截断回退。

当前验证仍属于单元测试和静态检查。由于完整 ASR 服务、Refiner checkpoint、GPU 环境和浏览器端到端测试没有在本次检查中全部启动，因此还不能据此宣称完成完整线上链路验证。

## 6. 当前尚未完成或需要修正的部分

这一节用于区分“设计中已有”与“代码中已实现”。以下内容在方向 A 计划中已经提出，但目前没有完全实现。

### 6.1 统一 ASR Hypothesis 协议尚未落地

计划中的 `system/contracts.py`、`ASRToken`、`ASRHypothesis` 和统一 sequence/audio timestamp 协议尚未建立。Whisper 和 Qwen3-ASR 仍主要通过现有 HTTP 返回文本，token 级 logprob、时间戳、no-speech probability 等字段没有统一提供。

### 6.2 置信度门控尚未实现

当前有可选的 `asr_confidence` 字段和会话记忆的置信度判断，但还没有真正实现 KEEP、DEFER、REFINE 三态决策、Hypothesis Tracker、综合门控分数、600 ms 调用节流、同文本去重、sequence 过期结果丢弃，以及与 always-refine/final-only 的对比实验。

当前离线流式模式的 latest-only 调度是“异步调度优化”，不能等同于研究方案中的“置信度门控”。

### 6.3 动态活动窗口尚未实现

目前的 `split_for_refinement()` 是长度和标点优先的安全分段，不是稳定前缀和 active span 方案。已提交前缀不可修改、只读左上下文、按停顿和稳定次数调整窗口、Whisper 滚动窗口合并以及最大回滚限制仍待开发。

### 6.4 实体保护链路还没有完全接通

保护模块本身有单元测试，但在线和批处理主链路仍需要补齐“掩码—精修—恢复—验证—回退”的闭环。当前代码主要做了术语 hint、显式别名标准化、部分实体审计，以及重复/截断回退。

需要继续修正：

- Web、批处理和本地 Qwen 流程都必须真正把掩码后的文本送入 Refiner，再统一恢复占位符。
- 批处理当前还没有把模型输出完整送入 `restore()` 并依据结果回退。
- 批处理结果中的 `refiner_accepted` 目前是默认写入 `True`，不能代表真实 Validator 结论。
- 本地 Qwen 流程需要在保护失败或审计失败时明确回退并写入拒绝原因。
- 旧的批处理 system prompt 仍保留 `<KEY>` 追加式约定，需要统一为独立 metadata 或结构化输出，避免污染评测文本。

### 6.5 完整 Validator 尚未实现

当前只有轻量的重复、截断、占位符和实体审计逻辑，尚未形成独立的 JSON Schema Validator、Protected-span Validator、Edit-scope Validator 和 Semantic Validator，也没有完整的数字、否定词、实体、代码和 unsupported addition 校验。

### 6.6 双参考数据和研究评测尚未完成

当前已有项目原有数据、AASR-Bench 入口和若干结果文件，但方向 A 所需的逐字参考与意图参考还没有正式建立。双参考标注、按说话人划分数据集、流式指标统一统计、replay_stream、跨后端和跨领域实验均待完成。

## 7. 后续工作安排

建议按照“先安全闭环，再研究能力，最后实验和论文”的顺序推进。

### 阶段 0：完成保护闭环和结果语义修正（最高优先级）

1. 抽象统一的 `protect → refine(masked) → restore → validate → fallback` 函数。
2. 接通 Web、批处理和本地 Qwen 流程，去除入口之间的行为差异。
3. 将模型输出失败、占位符异常、实体丢失、重复和截断分别记录为明确的 reject reason。
4. 修正 `refiner_accepted`，只由真实校验结果生成。
5. 更新批处理 prompt，移除旧的 `<KEY>` 用户文本污染方式。
6. 增加 fake Refiner 集成测试，覆盖正常输出、漏占位符、乱序占位符、未知占位符和异常重复。

验收标准：任一入口保护校验失败都返回原始 ASR 文本；三个入口的 JSONL 字段含义一致；集成测试证明一次错误精修不会覆盖原始实体。

### 阶段 1：统一协议和事件日志

- 新增 `system/contracts.py`，定义 session、sequence、text、language、final、tokens、confidence、backend 等字段。
- 修改 ASR 服务输出统一的 `ASRHypothesis`。
- 记录 ASR、门控、Refiner、Validator 分阶段延迟。
- 新增事件 JSONL schema 和版本号。
- 实现 `experiments/scripts/replay_stream.py`，允许不重新跑 ASR 即重放同一批事件。

验收标准：Whisper 和 Qwen3-ASR 的事件可以被同一套 tracker/gate 消费；replay 在相同配置下得到与在线流程一致的决策和输出。

### 阶段 2：实现规则门控

- 新增 `hypothesis_tracker.py`，计算 LCP、revision ratio、unchanged updates、tail age 和稳定前缀。
- 新增 `feature_extractor.py`，提取置信度、停顿、重复、口头语、自我修正、标点和实体风险。
- 新增 `gating.py`，实现 KEEP、DEFER、REFINE。
- 加入同文本去重、调用间隔、latest-only 和 sequence 过期结果丢弃。
- 与 always-refine、final-only 建立统一对比脚本。

验收标准：日志中能解释每次 Refiner 调用的原因，并能报告调用次数、接受率、回退率、p50/p95 延迟和文本质量。

### 阶段 3：动态 active span 和结构化 Refiner

- 新增 `span_planner.py`，区分 committed prefix、readonly context 和 active span。
- 自我修正发生时扩展到被修正内容，长句和长静音时安全截断。
- 将 Refiner 输出改为受约束 JSON：`decision`、`text`、`reason_codes`。
- 服务端计算实际编辑范围，不信任模型提供的字符偏移。
- 对 Whisper 滚动窗口增加稳定文本合并和最大回滚限制。

验收标准：已提交前缀不被普通精修修改；过期结果不会覆盖新结果；长文本不会因生成长度不足而出现重复或截断。

### 阶段 4：完整 Validator 和回退机制

- 实现 schema、保护片段、编辑范围和语义四类校验。
- 检查数字、日期、否定词、实体、URL、邮箱和代码的保真度。
- 记录 `accepted`、`fallback_raw`、`reject_reason` 和校验耗时。
- 在前端区分“模型没有修改”和“模型修改后被拒绝”。

验收标准：故意构造的实体替换、数字变化、否定反转、无依据扩写和大范围删除均能够回退并留下可定位原因。

### 阶段 5：双参考数据集和指标体系

- 先建立 1–2 小时起步集，再扩展到 5–10 小时正式集。
- 每条样本同时标注逐字参考和意图参考，并标记 filler、重复、自我修正、实体、数字、否定和领域。
- 按说话人划分 train/dev/test，测试集不用于阈值调优。
- 实现 CER/WER、意图保持、实体/数字准确率、错改率、unsupported addition、调用次数和流式延迟统计。

验收标准：能够在同一测试集上公平比较 Whisper only、Qwen3-ASR only、always-refine、final-only、rule-gate 和完整方案。

### 阶段 6：学习型门控、消融和论文实验

- 用开发集构造“精修有收益/无收益或有风险”的 gate 标签。
- 先训练 Logistic Regression 作为可解释基线，再评估 LightGBM 或小型 MLP。
- 做概率校准，搜索质量、延迟和调用成本的平衡点。
- 完成 confidence、stability、disfluency、dynamic span、protection、validator、DEFER 和 latest-only 消融。
- 做跨 ASR、噪声、中英混合、长对话和未见领域泛化实验。
- 输出错误类型分析、成功/失败案例、质量—延迟曲线和统计置信区间。

验收标准：最终结论来自固定测试集和可复现配置，阈值只在开发集确定，论文表格可以由脚本重新生成。

## 8. 建议的近期执行顺序

1. 先完成阶段 0，优先修正保护链路和 `refiner_accepted` 的真实语义。
2. 立即补齐 fake Refiner 集成测试，避免后续门控开发建立在不可靠的保护输出上。
3. 再实现统一协议、事件日志和 replay；没有 replay 就很难公平比较门控策略。
4. 然后实现规则门控和 tracker，先用可解释规则建立基线。
5. 在规则门控稳定后，再接入 active span、结构化输出和完整 Validator。
6. 最后建设双参考数据集，进行学习型门控和论文实验。

不建议现在直接开始学习型门控或大规模论文实验，因为统一输入协议、保护闭环和双参考标签还没有冻结，过早实验会导致结果不可复现或指标含义不一致。

## 9. 当前风险与注意事项

- 仓库当前存在未提交修改和新增文件；本报告描述的是工作区快照，不代表已经形成干净发布分支。
- 完整推理依赖模型 checkpoint、ASR 服务、CUDA/Transformers 环境和音频数据；本次验证没有替代这些依赖做端到端性能结论。
- 术语库属于用户验证数据，必须继续保持精确匹配、领域隔离、会话 TTL 和高置信晋级限制，不能将一次模型猜测自动写回永久词典。
- Refiner 的“清理口语”与“保持逐字内容”是两个不同目标，后续评测必须同时保留 verbatim 和 intended 两套指标。
- 任何门控阈值、窗口大小和调用次数目标都应通过开发集实验确定，不能直接把计划中的默认值写成实验结论。

## 10. 当前阶段的完成定义

当以下条件全部满足时，才可以认为方向 A 的 MVP 完成：

1. Whisper 和 Qwen3-ASR 都输出统一 Hypothesis 协议。
2. 系统具备可解释的 KEEP/DEFER/REFINE 规则门控。
3. 数字、日期、实体和代码经过真正的掩码、恢复和校验闭环。
4. Refiner 输出异常时自动回退原始 ASR 文本。
5. 已提交前缀和 active span 能够限制历史文本修改。
6. always-refine、final-only、rule-gate 三种策略可以在同一批 replay 数据上复现。
7. 至少拥有 intended CER、实体/数字准确率、调用次数和 p95 延迟四项可比较指标。

达到上述 MVP 后，再继续推进学习型门控、完整双参考数据和论文级消融实验。
