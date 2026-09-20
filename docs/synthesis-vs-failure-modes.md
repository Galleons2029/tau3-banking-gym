# 合成 pipeline 能否针对性覆盖 tau3-banking 的失败模式

配套阅读：[tau3-banking 失败模式分析](tau3-banking-failure-analysis.md)、[任务合成规格](task-synthesis.md)

## 0. 结论

**不能。** 当前 `src/tau3/synthesis/` 的 tau3-AA 路径在架构上**没有"针对性"这个概念**——没有任何配置项、CLI 参数或数据结构可以表达"多生成某一类任务"。更关键的是三个结构性阻断，使得即便加上配置开关，它也生不出打中主要失败模式的任务：

| 阻断 | 证据 | 后果 |
|---|---|---|
| **族别硬编码 + 强制均分** | [models.py:10](../src/tau3/synthesis/models.py#L10) `FAMILIES` 是 4 元 `Literal`；[scenarios.py:127](../src/tau3/synthesis/scenarios.py#L127) `family = FAMILIES[slot % 4]`；[workflow.py:98](../src/tau3/synthesis/workflow.py#L98) 强制 `num_tasks % 4 == 0` | 已发布的 200 题恰好 50/50/50/50。配比不可调 |
| **规则目录只覆盖 13 个工具 / 27 篇文档** | [catalog.py](../src/tau3/synthesis/catalog.py) 的 `tools` 白名单；`train-200-v3/catalog.json` 27 篇证据文档 / 域内 698 篇 | **失败动作的 60% 用到目录外的工具，今天完全无法表达** |
| **准入 = teacher 能做对** | [workflow.py:368](../src/tau3/synthesis/workflow.py#L368) `passed = reward==1 and reviewed.passed`；[workflow.py:447](../src/tau3/synthesis/workflow.py#L447) `accepted = any(2 trials passed)` | 难度天花板 = 教师模型能力。**最难的那一档被系统性筛掉** |

一句话总结这三条合起来的效果：**pipeline 生产的任务分布，恰好集中在被测模型已经做对的区间。**

### 分布对照（最直接的证据）

按参考动作数分桶，对比 benchmark 的 97 题与合成的 200 题：

| 参考动作数 | benchmark 97 题 | 该桶 d3note pass@1 | 合成 200 题 |
|---|---|---|---|
| 1–4 | 31% | 0.40 | 34% |
| 5–9 | 30% | 0.19 | 36% |
| 10–14 | 12% | 0.15 | **30%** |
| 15–19 | 13% | **0.02** | **0%** |
| 20+ | 13% | **0.00** | **0%** |

**模型崩溃最严重的 26% 的 benchmark，在合成集里占比为 0。** 合成任务的参考动作数上限是 13（`credit_limit` 族固定 13 个），众数是 3（63/200）。

准入数据同样说明问题：200 题里 187 题第一次采样就通过，教师模型两次 trial 全过的有 155/200。**这批任务对教师是"随手做对"的难度。**

## 1. pipeline 现状

仓库里其实有两条合成路径，能力边界完全不同：

### A. tau3-AA（`tau3 synthesize generate/validate/export`）

固定 `banking_knowledge` 世界，知识库与业务工具不动，只合成客户、账户状态、目标和参考动作。四个族：

| 族 | 业务逻辑 | 求解器 | 参考动作数 |
|---|---|---|---|
| `selection` | 三选一挑最便宜的合规支票账户 | [`solve_selection`](../src/tau3/synthesis/scenarios.py#L36) | 3 |
| `cashback` | 找出返现算错的交易并逐笔争议 | [`reward_points`](../src/tau3/synthesis/scenarios.py#L81) | 5–11 |
| `credit_limit` | 提额资格判定（6 种拒绝分支） | [`cli_decision`](../src/tau3/synthesis/scenarios.py#L56) | 13 |
| `ordering` | 归集余额→关户（可选附带开企业户） | 无（直接构造） | 5–13 |

**这条路径最大的资产是"程序化求解 + 独立复核"**，不是 LLM 猜答案：

- 参考动作由确定性求解器算出，规则值从 KB 用正则抽取并带 sha256 证据（[catalog.py](../src/tau3/synthesis/catalog.py)，抽不到就 fail closed）。
- [`validate_static`](../src/tau3/synthesis/validation.py#L389) 跑严格真实工具重放 + 两次重放哈希一致性 + benchmark 评测器必须给参考轨迹打满分。
- [`counterexamples`](../src/tau3/synthesis/validation.py#L341) 构造反例并要求它们**必须拿不到满分**：`empty`、`missing_final_action`、`extra_write`、`wrong_valid_product`、以及对 `account_class`/`new_rewards_earned`/`new_credit_limit`/`amount`/`denial_reason` 的逐字段扰动。

最后这条尤其值得强调：**`extra_write` 反例正是失败模式 F4（过度动作）的判别器**，pipeline 已经在为每一道题验证"多做一次写操作会被判负"。判别能力是有的，缺的是把它当成训练目标去覆盖。

### B. worldgen v2 + `world-tasks`

`src/tau3/worldgen/v2/` 合成**全新的世界**（`banking_synth` 域）：实体、产品、政策、操作、场景都由 spec 描述（[specs.py](../src/tau3/worldgen/v2/specs.py)），`ScenarioSpec.kind` 支持 `selection | service | ordering | denial | grounding`。

但 `tau3 synthesize world-tasks` 这一阶段**只做"表达"，不做"业务"**——[`expression_errors`](../src/tau3/synthesis/world_tasks.py#L51) 明确禁止改动数字字面量和机器标识符，它把已准入的种子场景重写成不同措辞的任务文本。业务结构全部来自 V2 world spec。

**它是新世界，不是 banking_knowledge。** 用它造的任务不会测到本次分析所针对的那套 KB、那套 discoverable 工具、那套转接码。如果目标是"修模型在 tau3-banking 上的表现"，B 路径是另一件事。

## 2. 逐失败模式的可表达性

对照失败分析的 8 类模式，评估 tau3-AA 今天的表达能力：

| 失败模式 | 今天能否合成 | 依据 |
|---|---|---|
| **F3 枚举不全** | ✅ **已经在生产** | `cashback` 族就是这个结构：`count = rng.randint(2,5)` 笔交易，其中 N−1 笔返现算错、至少留 1 笔正确做干扰项（[scenarios.py:220](../src/tau3/synthesis/scenarios.py#L220)）。参考动作要求逐笔争议 |
| **F3b 用户工具交接参数错** | ✅ **已经在生产** | 同上，`give_discoverable_user_tool` + 逐笔 `call_discoverable_user_tool`，requestor=`user` |
| **F4 过度动作污染 DB** | ✅ 判别器已有，⚠️ 未作为目标 | `extra_write` 反例每题必验；但没有任何任务**刻意**诱发重复写（如用户中途复述需求） |
| **F2 分级规则套错档** | ⚠️ **部分**：只有 CLI 三档 | `cli_tiers` 覆盖 entry/mid/premium 的 age/utilization/months/max_increase。但 Reg-E 责任上限、临时信用资格、APY 叠加**全都不在目录里** |
| **F1 KB 工具发现失败** | ⚠️ **强度不足** | 只有 13 个工具入目录，且 `credit_limit` 族的 5 个只读工具是固定序列，模型见一次就能记住；真实失败集中在 `get_bank_account_transactions_9173`、`get_debit_cards_by_account_id_7823` 这些**目录外**工具 |
| **F5 过早升级转人工** | ❌ | `transfer_to_human_agents` 不在目录，且**没有任何"不该转"的反例**。`counterexamples` 里没有 `premature_transfer` 这一项 |
| **F6 转接 reason 选错** | ❌ | 同上。`TransferReasonLiteral` 枚举在域内定义，KB 里有对应文档，但目录没抽 |
| **F7 凭空增参数** | ❌ | `tool_signatures` 被抽进目录了，但只用于 `validate_references`，没有生成"签名相近易混淆"的任务 |
| **F8 完全不作为** | ❌ | 没有会诱发放弃的任务形态 |

### 60% 的失败动作今天无法表达

把 485 条轨迹里全部 2214 个未匹配的期望动作，按其工具是否在目录内分类：

```
目录内（今天可合成）    889   40%
目录外（今天不可合成） 1325   60%
```

目录外的 top：

```
167  get_bank_account_transactions_9173
166  file_credit_card_transaction_dispute_4829
125  get_debit_cards_by_account_id_7823
121  file_debit_card_transaction_dispute_6281     ← Reg-E，F2 最大来源
 79  order_debit_card_5739
 70  close_debit_card_4721
 61  unfreeze_debit_card_3893
 56  get_card_last_4_digits
 46  freeze_debit_card_3892
 39  apply_for_credit_card
 39  submit_interest_discrepancy_report_7294      ← APY，F2 第二来源
 38  apply_checking_account_credit_5829
 36  transfer_to_human_agents                     ← F5/F6 全部
```

**整个借记卡业务线（争议、订卡、冻结/解冻、PIN）在目录里是空白**，而它恰好是 dispute 族（189 条轨迹，pass 0.12）的主体。

## 3. 三个阻断的性质：哪些是硬的，哪些是软的

### 阻断 1：族别硬编码 —— 软，改动小

`FAMILIES` 是 `models.py` 里的常量元组 + `ScenarioSkeleton.family` 的 `Literal`，`sample_candidate` 用 `slot % 4` 分配。要支持配比，需要：

- `SynthesisConfig` 增 `family_weights: dict[str, int]`（注意 `model_config = ConfigDict(extra="forbid")`，且 config 整体进 manifest 并参与 `--resume` 的一致性校验，所以这是个 breaking change，旧 bundle 无法 resume）。
- 放宽 [workflow.py:98](../src/tau3/synthesis/workflow.py#L98) 的 `% 4` 约束，同步改 [`grouped_split`](../src/tau3/synthesis/workflow.py#L466)（按族各取 20% 做验证集）和 `sampling_requirements`。

**工作量：小。** 这一条不是真正的障碍。

### 阻断 2：目录只有 13 工具 —— 中等，但路是通的

好消息是**域里工具齐全**（`tools.py` 有 60+ 个业务工具，上面列的缺失项全部存在），而且**规则在 KB 里是可正则抽取的**，形态和现有目录条目一致。以 Reg-E 为例，`doc_bank_accounts_bank_accounts_(general)_031`：

```
- Reported within 2 business days of statement: Maximum liability $50
- Reported within 60 days of statement: Maximum liability $500
- Reported after 60 days: Unlimited liability

Dispute the earliest (first) transaction when multiple duplicates exist.
...
4. Customer cannot exceed the maximum open disputes for their checking account tier:
   Entry Tier max 2, Mid Tier max 3, Premium Tier max 4, Elite Tier max 5.
```

这段同时给出了**分档规则**（F2）、**"多笔重复取最早一笔"的选择规则**（F3）和**每账户档位上限**（F2 边界）。抽取难度与现有的 `cli_tiers` 正则相当。

需要新增的是**求解器**：
- `liability(discovery_date, transaction_date) -> 50 | 500 | -1`
- `provisional_credit_eligible(...)` —— 依赖 `written_statement_provided` 等，需要先确认 KB 里的判定条件是否完备
- `card_action(dispute_category, card_in_possession) -> keep_active | freeze_pending_investigation | close_and_reissue`

**风险点**：`file_debit_card_transaction_dispute_6281` 有 18 个参数，工具本身**只校验枚举合法性，不校验业务正确性**——它会接受错误的 `customer_max_liability_amount` 并返回成功。这意味着求解器一旦写错，`validate_static` 的重放不会报错，错误会静默进入参考答案。所以这一族必须配套逐字段的反例（现有 `counterexamples` 的字段白名单要扩展到 `customer_max_liability_amount` / `provisional_credit_eligible` / `card_action` / `dispute_category`）。

**工作量：中。** 每个新族约 = 目录抽取（~40 行）+ 求解器（~30 行）+ `sample_candidate` 分支（~80 行）+ `check_business` 分支（~20 行）+ 反例字段（~5 行）。`scenarios.py` 的 `sample_candidate` 已经是 430 行的单函数，再加两族需要先拆分。

### 阻断 3：teacher-solvable 准入 —— 硬，是设计取舍

这是最根本的一条。[workflow.py:447](../src/tau3/synthesis/workflow.py#L447) `accepted = any(trial passed)`，配合 [generate_bundle](../src/tau3/synthesis/workflow.py#L95) 的 `candidate_attempts=10` 重采样：**教师做不出来的业务骨架会被丢弃并重新采样，直到采到一个教师能做对的。** 200/200 发布、0 拒绝，就是这个机制的结果。

这个设计对 **SFT** 是正确的——[`collect_sft`](../src/tau3/synthesis/workflow.py#L586) 要采集教师的成功轨迹，题目必须可解。但它同时意味着：

> **你无法用这条 pipeline 生产"比教师更难"的题。** 而失败分析显示，被测模型的崩溃点在 15+ 动作的长程任务上；如果教师（`GLM-5.3-Flash`）在那个区间也不稳，这些题一道都出不来。

这不是 bug，是 SFT 导向和评测导向的目标冲突。要生产针对性的**评测**集或 **RL** 集，需要把"可解性"与"教师能解"解耦：

- 可解性由 `validate_static` 保证（参考轨迹重放 + 评测器满分 + 反例判负）——**这一层已经完全够用，不需要任何 LLM 参与**。
- 教师 trial 应降级为**难度标签**而非准入门槛：记录 `teacher_pass_rate`，按目标难度分层采样，而不是一票否决。
- `run_trial` 已经在记录 `retrieval_calls` / `evidence_coverage` / `visible_context_chars`，这些是现成的难度代理指标，目前完全没用于准入或分层。

## 4. 还缺的两块能力

### 4.1 判分维度单一

[scenarios.py:551](../src/tau3/synthesis/scenarios.py#L551) 写死 `reward_basis=["DB"]`，[validation.py:118](../src/tau3/synthesis/validation.py#L118) 还会显式拒绝非 DB 判分。这意味着合成任务无法表达：

- `forbidden_actions` —— **F5（不该转人工）、F4（不该重复写）最自然的表达方式**
- `nl_assertions` —— 需要模型说出某个推理依据（benchmark 的 task_102 就是这种）
- `communicate_info` —— 需要模型向用户传达某个事实

其中 `forbidden_actions` 是性价比最高的：`Task` schema 已经支持，评测器已经支持，只需要 `sample_candidate` 填上、`validate_static` 补一条"参考轨迹不得触发 forbidden"的检查。

### 4.2 用户模拟器行为不可控

用户侧完全由 `StructuredUserInstructions` 的自然语言字段驱动（`reason_for_call` / `known_info` / `task_instructions`）。失败模式 F5/F8 的触发条件是**用户的行为压力**——benchmark 的 task_008 明确写了"你必须极度坚持，被拒绝就换下一个 offer，三个都被拒就循环回来再争"。合成侧没有这种行为脚本的结构化表达，只能靠往 `facts["interaction"]` 里塞自由文本，而这段文本还要过 LLM 改写和 `no_answer_leak` 复核。

## 5. 建议路线

按"能打中的失败模式 / 改造成本"排序：

### 第一步：不改架构，先把已有族别用足（1–2 天）

1. **加 `family_weights`**，把 `cashback` 拉到 50%+。它已经是 F3/F3b 的正确结构，只是配比被锁死在 25%。同时把 `count = rng.randint(2,5)` 的上限提到 8–10，直接压长程枚举能力。
2. **`ordering` 族的 `count`（[scenarios.py:443](../src/tau3/synthesis/scenarios.py#L443)）上限从 3 提到 6**，`open_business` 从 1/20 提到 1/3。`ordering` 是唯一天然可以线性拉长参考动作数的族（每个源账户 = 2 个写动作），把它推到 15–20 动作，就能补上当前 0% 的那个桶。
3. 这一步能覆盖 F3 / F3b / F4，以及 F1 的长程部分。

**前提**：必须先确认教师在拉长后的任务上还能通过，否则第 3 节的阻断会直接把它们全部筛掉。建议同时做第二步的 (1)。

### 第二步：解耦准入与教师能力（2–3 天）

1. 把 `accepted` 的定义从 `any(trial passed)` 改成 `static_passed and text_checked`，教师 trial 结果降级为 `metadata.jsonl` 里的 `teacher_pass_rate` 标签。
2. `export` 按目标难度分布采样（用 `teacher_pass_rate` + `reference_actions` 分层）。
3. `collect-sft` 只从 `teacher_pass_rate > 0` 的子集采集——**SFT 的可解性要求依然满足，但评测/RL 集不再被削顶**。

这一步是后续所有针对性合成的前置条件。

### 第三步：扩目录，补借记卡与转接两条线（1–2 周）

优先级按失败动作数：

| 新族 | 覆盖失败模式 | 目录需抽 | 求解器 |
|---|---|---|---|
| `debit_dispute` | F2（Reg-E 分档，121+166 个失败动作）、F3（"取最早一笔"） | `(general)_031/036/037` + 档位上限 | liability / provisional_credit / card_action |
| `escalation` | F5 + F6（36 个失败动作，但影响 5 个全灭任务） | `TransferReasonLiteral` + 各 reason 的 KB 场景文档 | reason 分类；**需配 `forbidden_actions`** 表达"不该转" |
| `interest_discrepancy` | F2（APY 叠加，39 个失败动作） | 各账户 base APY + linked-checking boost + card boost | APY 累加 + 利息差额 |

其中 `escalation` 族必须先做 4.1 的 `forbidden_actions` 支持——否则"该转却没转"能判，"不该转却转了"判不了，而后者才是实际发生的失败（27 次无端转人工）。

先做 `sample_candidate` 的拆分（每族一个模块），否则第 430 行的单函数会继续膨胀。

### 不建议做的

- **不要走 worldgen v2 路径去解这个问题**。它造的是 `banking_synth` 新世界，与本次分析的 `banking_knowledge` KB、discoverable 工具集、转接码都不共享。它有独立的价值（世界多样性），但不是 tau3-banking 失败模式的解药。
- **不要用 LLM 直接生成参考动作**。现有的程序化求解 + 反例验证是这套 pipeline 最值钱的部分；F2 类失败的本质就是"政策算术算错"，如果参考答案本身由 LLM 算，等于用一个会犯同样错误的系统去标注。

---

*本文结论基于：`src/tau3/synthesis/`（3599 行）、`src/tau3/worldgen/v2/`（5605 行）源码阅读，已发布 bundle `data/synthetic/tau3-aa/train-200-v3/`（200 题 metadata 全量统计），以及 `data/simulations/tau3_banking_dots3_max/` 485 条轨迹的失败动作与目录工具的交叉比对。*
