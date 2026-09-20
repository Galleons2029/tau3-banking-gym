# τ-Banking 同分布合成 Pipeline 设计

> 这是旧版设计目标，不表示既有产物已经复现论文。当前实现、发布门禁和使用方式见 [Worldgen V2](worldgen-v2.md)。V2 对正确性、自然语言审核与行为接近程度分别记录证据，未验证的性质保持 `INCONCLUSIVE`。

目标：复现论文 §4 的 structured-to-unstructured 生成流程，产出与官方 τ-Banking 种子文档 + 种子数据库**统计同分布、难度同量级、可自动验证**的合成语料。

---

## 0. 标定目标（从官方统计反推）

任何"同分布"声明都必须先有可测的靶子。论文 Table 1 + Appendix 给出的硬指标：

| 维度 | 官方值 | 派生约束 |
|---|---|---|
| 文档数 | 698 | — |
| 总 token (cl100k) | 194,562 | — |
| 平均 token/文档 | 278.7 | 长度分布均值锚点 |
| 知识类别 (category) | 21 | — |
| 主题 (topic/feature) | 71 | ≈3.4 topic/category，≈9.8 doc/topic |
| 可发现工具 | 51 | ≈0.72 tool/topic（并非每个 topic 都有工具） |
| 常驻 agent 工具 | 14 | — |
| 任务数 | 97 | — |
| 平均 gold 文档/任务 | 18.6 | 1,804 个 gold 槽位 → 每篇文档平均被 2.6 个任务引用，**gold 集重叠度高** |
| 平均工具调用/任务 | 9.52 | 总计 ≈923 次 |
| 工具调用范围 | 1 – 33 | 右偏长尾分布 |

**行为级靶子**（比文本统计更重要，见 §9）：

| 指标 | 官方值 |
|---|---|
| 最佳 pass^1 | 25.52 (GPT-5.2 high + terminal) |
| Gold 配置最佳 pass^1 | 39.69 (Opus high) |
| Gold vs 检索平均差值 | −13.0 ~ −15.3 |
| no-knowledge pass^1 | ≈2%（其中 2 个任务是 grounding check） |
| long-context pass^1 | ≈12% |
| 文档召回率区间 | 28%（GPT-5.2 none）~ 62%（Opus + terminal） |
| 用户模拟器 task-critical 错误率 | 4/194 ≈ 2% |

如果合成语料在这几项上落在官方 ±3 个百分点内，才算真的"同分布"；文本长度对齐只是必要条件。

---

## 1. 总体架构

```
A. 结构化层生成      → schema/{categories,features,variables}.yaml
        ↓ (CSP 求解赋值 + 陷阱注入)
B. 工具层生成        → registry/tools.yaml  (14 permanent + 51 discoverable)
        ↓ (工具↔文档双向绑定)
C. 文档规划          → plan/allocation.json  (变量 → 文档 二部图)
        ↓ (原型 × 长度预算 × 冗余策略)
D. 占位符渲染        → render/*.md.tmpl  ([[var]] 未填充)
        ↓ (确定性填充 + 4 模型风格轮转)
E. Linter 套件       → 9 类自动检查，失败即回炉
        ↓
F. 任务 + DB 生成    → tasks/*.yaml + db/seed.json
        ↓
G. 可解性 & gold 最小性证明（符号执行 + 集合覆盖）
        ↓
H. 同分布验收门（文本统计 + 检索难度 + agent 行为）
        ↓
I. 增量刷新回路（变量变更 → 只重渲受影响文档 → 只重验受影响任务）
```

A–E 全自动，F 半自动（人写任务骨架、LLM 补全对话流），G–H 全自动，I 由 A–H 的内容哈希驱动。

---

## 2. Stage A：结构化层

### A1. 三级 schema

论文的关键洞察是"结构化知识库可以视为一个约束系统：每个变量是产品空间的一个维度"。把这一点显式落成 schema：

```yaml
# schema/categories.yaml   (21 条)
- id: cat_business_checking
  display_name: "Business Checking Accounts"
  kind: product            # product | protocol | program | support
  audience: both           # customer | internal | both
  topic_budget: 4          # 该类别下 feature 数，总和 = 71

# schema/features.yaml     (71 条)
- id: feat_navy_blue_business_checking
  category_id: cat_business_checking
  entity_name: "Navy Blue"
  entity_kind: account
  doc_budget: 10           # 该 feature 展开的文档数，总和 = 698
  variables: [var_nb_monthly_fee, var_nb_min_balance, ...]

# schema/variables.yaml
- id: var_nb_monthly_fee
  feature_id: feat_navy_blue_business_checking
  name: monthly_maintenance_fee
  type: currency           # currency|percent|int_days|int_count|enum|bool|duration|string
  value: 0.0
  visibility: customer_facing   # customer_facing | internal_only
  volatility: static            # static | promotional
  window: null                  # promotional 时为 {start, end}
```

命名分布要点（照抄官方观感）：
- 银行名固定 `Rho-Bank`
- 产品命名以**颜色 + 金属层级**为主轴：Blue / Sky Blue / Navy Blue / Cobalt Blue / Bluest / Green / Light Green / Evergreen / Purple；Bronze / Silver / Silver Plus / Gold / Platinum / Diamond Elite；外加少量概念名 Crypto-Cash Back / EcoCard
- 同一色系跨类别复用（Silver 既是储蓄账户层级又是信用卡层级）——这是**制造检索歧义的核心机制**，务必保留

### A2. 约束求解式赋值

不要让 LLM 直接填数值。先由 LLM 生成变量**骨架**（名称、类型、合理量纲区间），再用 CP-SAT / z3 / 拒绝采样求解具体值，约束来自三类：

1. **领域合理性**：`0 ≤ apy ≤ 6%`、`business_checking.min_balance ∈ [0, 25000]`、层级单调性 `apy(Bronze) < apy(Silver) < apy(Gold)`
2. **任务唯一性**（对应论文"the only savings account with APY below 3%, early direct deposit, and no overdraft fees must be Sky Blue"）：
   ```
   ∀ task t: |{ x ∈ Products : ⋀ c∈constraints(t) c(x) }| == 1
   ```
3. **陷阱约束**（对应论文四类失败模式，见 §7.2 表）：
   - *促销陷阱*：`base_apy(A) − base_apy(B) > promo_boost(B)`，让"追促销"的 agent 必错
   - *近似干扰项*：对每个任务，至少存在 2 个产品满足 |constraints|−1 条约束
   - *时效陷阱*：同一 feature 挂 2 组 promotional 变量，窗口一早一晚，且推荐结论相反；agent 必须先调 `get_current_time()`

求解失败时回退：放松最弱的领域约束，重跑；连续 3 次失败则报告给人工，说明 feature 空间维度不足。

### A3. 独立性原则

论文明确写了"each feature is defined independently in the structured database, and interactions between features are introduced only when required by downstream tasks"。实现上：

- 生成期 feature 之间**无引用**（`variables` 只含本 feature 的变量）
- 跨 feature 交互只能通过 `interactions.yaml` 显式声明，且必须挂到一个 task_id 上：
  ```yaml
  - id: itx_close_blocks_business_open
    required_by: [tau_task_042]
    predicate: "any(acct.status == CLOSED for acct in user.accounts) => deny(open_business_checking)"
    surfaced_in_docs: [doc_biz_checking_eligibility, doc_internal_closure_protocol]
  ```
- 未被任何 task 引用的 interaction 一律删除。这条纪律是"minimizes unintended collisions"的落地方式。

---

## 3. Stage B：工具层

### B1. 注册表

```yaml
- id: open_bank_account_4821
  base_name: open_bank_account
  suffix: 4821
  scope: agent_discoverable      # permanent | agent_discoverable | user_discoverable
  args:
    - {name: user_id, type: string, required: true}
    - {name: account_type, type: enum, required: true,
       values: [checking, savings, business_checking, business_savings]}
    - {name: account_class, type: string, required: true}
  reads:  [users, bank_accounts]
  writes: [bank_accounts]
  preconditions:
    - "not any(a.status == 'CLOSED' for a in accounts(user_id))  # 仅 business_*"
    - "account_age_days(oldest_open(user_id)) >= 14              # 仅 savings"
  effects:
    - "bank_accounts.insert(...)"
  doc_refs: [doc_0417]           # ≥1，由 Stage C 回填
```

### B2. 后缀分布

官方工具名一律是 `base_name_XXXX`，4 位数字。观察到的一个重要细节：**后缀会跨 base_name 碰撞**（`7291` 同时出现在 `transfer_funds_between_bank_accounts_7291`、`order_replacement_credit_card_7291`、`get_user_dispute_history_7291`）。所以：

```python
suffix = rng.randint(1000, 9999)     # 独立均匀采样，不做去重
assert (base_name, suffix) 唯一       # 只保证全名唯一
```

后缀的作用是切断参数化记忆——agent 不可能猜出 `8293`，必须真的检索到文档。合成时保留这个性质，并**每次重新生成语料时重掷后缀**，可作为防污染手段。

### B3. 工具↔文档绑定

- 51 个 discoverable 工具，每个至少被 1 篇文档提及，签名以**函数名 + 参数清单**形式内联（照 Appendix A.2 的写法：`Tool arguments:` 后接 bullet 列表，标注类型和 required）
- 约 60% 的工具签名写在 `internal_protocol` 文档的步骤里（"Use the get_closure_reason_history_8293 tool to..."），40% 写在专门的 `tool_doc` 文档里
- 反向：不存在文档提及但注册表里没有的工具（Linter L9）
- 14 个 permanent 工具建议集合：`KB_search`/`shell`、`get_current_time`、`get_user_information_by_email`、`get_user_information_by_phone`、`log_verification`、`get_credit_card_accounts_by_user`、`get_bank_accounts_by_user`、`unlock_discoverable_agent_tool`、`call_discoverable_agent_tool`、`give_discoverable_user_tool`、`list_unlocked_tools`、`transfer_to_human_agents`、`send_message_to_user`、`think`

---

## 4. Stage C：文档规划（变量 → 文档 二部图）

这是决定"同分布"成败的一步，也是官方 pipeline 里 "An LLM then allocates subsets of variables to document titles" 那一步。

### C1. 文档原型与长度预算

| 原型 | 占比 | 篇数 | 均值 tok | 说明 |
|---|---|---|---|---|
| `product_overview` | 10% | 70 | 420 | "X (Checking) at a Glance"：要点 bullet + Fees and Limits 表 + Important Details |
| `faq` | 20% | 140 | 240 | "Blue Account FAQ"：Q/A 对，3–6 组 |
| `internal_protocol` | 12% | 84 | 520 | "Internal: Credit Card Retention Protocol"：编号步骤 + 条件分支 + 工具签名 |
| `tool_doc` | 9% | 63 | 330 | "Applying Resolved Cash Back Dispute Corrections (Internal)"：Overview / Required Steps / Compliance |
| `program_terms` | 9% | 63 | 260 | 推荐奖励、返现规则，含资格与上限 |
| `howto_short` | 25% | 174 | 150 | "How do I view my monthly cashback?" |
| `eligibility_matrix` | 8% | 56 | 300 | 层级映射表（Entry/Mid/Premium Tier ↔ 等待期） |
| `promo_notice` | 7% | 48 | 165 | 带日期窗口的促销公告 |

加权均值 = 278.6 ≈ 官方 278.7。篇内长度取截断对数正态 `LogNormal(ln(μ)−σ²/2, σ=0.35)`，截断到 `[0.45μ, 2.2μ]`。

标题风格池（每个原型一套模板 + LLM 改写）：
- `{Entity} at a Glance` / `{Entity} Overview` / `{Entity} FAQ`
- `Internal: {Process} Protocol` / `{Process} (Internal)`
- `How do I {verb} my {noun}?` / `What happens if I {verb}?`
- `{Program} Referral Program` / `{Entity} and {Entity} Promotion`

### C2. 分配算法

对每个 feature，设变量集 V，文档预算 d（来自 `doc_budget`）：

```
1. 按 visibility 划分 V = V_cust ∪ V_int
2. 抽取 d 个标题（原型比例按全局配额分层采样，internal_* 原型只吃 V_int）
3. 求解一个带约束的二部匹配：
   - 覆盖：∀v ∈ V, deg(v) ≥ 1
   - 碎片化：deg 的期望 = 1.35（35% 变量出现两次，其中一次为交叉引用桩）
   - 容量：|vars(doc)| ~ 依原型而定，overview 6–12，faq 3–6，howto_short 1–2
   - 反聚合：同一任务所需的变量不得全部落入同一篇文档
     ∀ task t, ∀ doc D:  |vars(t) ∩ vars(D)| ≤ ⌈|vars(t)| / 3⌉
```

最后一条约束直接决定 gold 文档数。官方 18.6 篇/任务意味着任务知识被切得非常碎——把 `⌈|vars(t)|/3⌉` 作为调参旋钮，先跑一遍任务集，测得 gold 均值后二分调整。

**目标分解参考**（18.6 = 3–4 个产品/protocol feature × 每个 3–4 篇 + 5–9 篇工具文档 + 2–3 篇资格/促销文档）。

### C3. 冗余与交叉引用

Figure 2 第三阶段的 "review, edit, link, and de-duplicate" 是这样落地的：

- 一个变量至多出现在 2 篇文档
- 第 2 次出现有两种形态，按 7:3 抽样：
  - **一致重复**：值完全相同（制造检索冗余，降低单点失败）
  - **交叉引用桩**：`A: See the "referral bonus for Blue" document for details.` — 引用目标必须存在（L6），且强制多跳
- 近重复检测：MinHash (shingle=5) Jaccard > 0.72 或 dense cosine > 0.90 → 强制把后者改写成交叉引用桩

---

## 5. Stage D：渲染

### D1. 两段式占位符（这是保证事实一致性的关键）

论文 Appendix A 明确提到 "documents may contain templated variables (shown as `[[variable_name]]`) that are populated with concrete values during finalization"。照做：

**Pass 1（LLM）**：只给标题 + 分配到的变量的**名称/类型/单位**，**不给值**，要求输出中所有数值一律写成 `[[var_name]]`。

**Pass 2（确定性）**：从 `variables.yaml` 取值，按类型格式化（currency → `$1,250`，percent → `2.5`，int_days → `3`）后替换。

**硬门禁**：Pass 1 输出中出现任何未包裹在 `[[ ]]` 里的阿拉伯数字（白名单：步骤编号、年份格式、条款编号）→ 判定为幻觉，直接重生成。这一条挡掉了 90% 的事实漂移。

### D2. 风格轮转

论文用 GPT-5 / GPT-5.2 / Claude-4.5-Opus / Gemini-3-Pro 四模型轮转以制造措辞与结构多样性。合成时给每篇文档分配一个三元组：

```
(model, persona, register)
persona ∈ {product marketing, compliance officer, support-ops trainer, help-center writer}
register ∈ {customer-facing warm, customer-facing terse, internal procedural, internal legalistic}
```

约束：`internal_protocol` / `tool_doc` 只能配 internal register；同一 feature 下的文档不得全部同一 model（避免风格聚簇被 embedding 轻易聚类，那会人为降低检索难度）。

### D3. Stage-1 / Stage-2 prompt 模板

**Stage 1（变量骨架）**

```
你在为一家虚构的美国零售银行 Rho-Bank 设计产品目录的结构化 schema。

类别：{category.display_name}
该类别下的产品/流程：{entity_name}

请输出该 entity 的 8–16 个属性变量，每个变量给出：
- name（snake_case）
- type（currency | percent | int_days | int_count | enum | bool | duration | string）
- unit
- plausible_range（不要给具体值，只给区间；enum 给候选集）
- visibility（customer_facing | internal_only）

只输出 YAML，不要解释。不要输出任何具体数值。
```

**Stage 2a（标题分配）**

```
以下是 {entity_name} 的属性变量清单：
{variables with name/type/unit, 无值}

请生成 {d} 个真实客服知识库中会出现的文档标题，并把上述变量分配给标题。
要求：
- 标题风格需覆盖：产品总览、FAQ、单点 how-to、内部流程、资格矩阵、促销公告
- 每个变量至少被分配一次
- 单点 how-to 类标题只分配 1–2 个变量
- 不要把 {task_critical_vars} 中的变量分配到同一个标题下

输出 JSON：[{"title": ..., "archetype": ..., "variables": [...]}]
```

**Stage 2b（正文渲染）**

```
你是 Rho-Bank 的 {persona}，正在撰写一篇内部知识库文档。

标题：{title}
文档类型：{archetype}
目标长度：约 {target_tokens} tokens
语域：{register}

可用变量（只有这些，不得引入其他数值事实）：
{var_name}: {type}, 单位 {unit}
...

{if tool_refs}
必须在正文中以如下形式提及以下工具及其完整参数清单：
{tool signature block}
{endif}

{if crossref_stubs}
以下信息不要展开，改为引用其他文档：
- {var} → 写成 See the "{target_title}" document for details.
{endif}

硬性要求：
1. 所有数值必须写成 [[variable_name]] 占位符，绝对不要写出任何具体数字
2. 不要提及本文档未列出的产品、费用、工具
3. 使用 Markdown；表格用于 fees/limits；内部流程用编号步骤
4. 直接输出文档正文，不要前言后语
```

---

## 6. Stage E：Linter 套件

每篇文档过 9 关，任一失败即回炉（最多 3 次），3 次仍失败则标记人工。

| ID | 检查 | 实现 |
|---|---|---|
| L1 | 数值溯源 | 渲染后正文中的每个数值必须能反查到一个已分配变量；否则 fail |
| L2 | 跨文档矛盾 | 同一 `variable_id` 在不同文档渲染值不一致 → fail（promotional 变量按窗口豁免） |
| L3 | 工具签名一致 | 文档里写的参数名/类型/required 与 `tools.yaml` 逐字段比对 |
| L4 | 任务唯一性 | 对每个 task 的约束合取在结构化 DB 上求解，`|解| == 1` |
| L5 | 单文档泄漏 | 单篇文档暴露的 task-critical 变量数 ≤ ⌈|vars(t)|/3⌉（防单跳捷径） |
| L6 | 悬空引用 | 交叉引用桩的目标标题存在且唯一匹配 |
| L7 | 长度整形 | 落在 `[0.45μ, 2.2μ]` 内；全局均值 278.7 ± 4 |
| L8 | 近重复 | MinHash / dense cosine 阈值检测，超阈 → 转交叉引用 |
| L9 | 孤儿工具 | 每个 discoverable 工具被 ≥1 篇文档提及；文档提及的工具都在注册表内 |

L4 和 L5 是"unit-style tests"的具体形态——论文说"constraints can be validated directly against the structured database, allowing unit-style tests to ensure that new tasks are well-posed and non-conflicting as the task count grows to the hundreds"，就是这两条。

---

## 7. Stage F：任务与数据库生成

### F1. 任务 schema

```yaml
task_id: tau_task_042
archetype: topological_ordering
persona:
  name: Jordan Chen
  age: 36
  city: Seattle, WA
  occupation: marketing consultant
verification:                  # 2-of-4 规则：dob / email / phone / address
  name: Jordan Chen
  user_id: jc61f7a8d2
  dob: "03/15/1988"
  email: jordan.chen@consulting.io
  address: 4521 Pine Street, Seattle, WA 98101
sim_date: "11/14/2025"         # 供 get_current_time() 返回，驱动促销时效
initial_db_patch:              # 打在 base seed DB 上的 diff
  bank_accounts:
    - {id: 61a8b7c6d5e4f321, class: Evergreen, type: checking,
       balance: 3500, opened: "11/02/2025"}   # 12 天 → 触发 14 天租期约束
constraints:                   # 用户在对话中逐步透露
  - {reveal_at: opening,  predicate: "monthly_maintenance_fee == 0"}
  - {reveal_at: turn_3,   predicate: "min_balance_requirement == 0"}
hidden_dependencies:
  - "close(any) ⇒ status CLOSED ⇒ block open_business_checking"
  - "open_savings requires oldest_open_account_age_days >= 14"
expected_actions: [...]        # 有序工具调用序列
target_final_db_state: {...}
gold_documents: [doc_0117, doc_0233, ...]     # 目标 |gold| ~ 18.6
flow_rules:
  - trigger: agent_proposes_reorder
    user_says: "Wait, why can't we just do the closures first?"
  - trigger: agent_explains_dependencies
    user_says: "Oh wow, I had no idea! Okay, you're the expert."
```

### F2. 任务原型分布（97 条）

按论文 §7.2 的失败模式聚类反推配额：

| 原型 | 配额 | 对应失败模式 | 生成要点 |
|---|---|---|---|
| A. 多约束产品推荐 | 20 | 复杂互依（~14.5%） | 促销 boost < 基准差；3+ 近似干扰项；要求"只给一个答案" |
| B. 流程协议执行 | 18 | — | 严格 step 顺序 + 跳过条件（如已有 retention 记录则跳过挽留） |
| C. 拓扑排序依赖 | 14 | 隐式子任务顺序（~5%） | 用户按错误顺序提 3–4 件事；正确顺序需跨 2+ 文档推断 |
| D. 过度信任探针 | 10 | 轻信用户断言（~4%） | 用户声称"我的争议都批了"，DB 里仍是 UNDER_REVIEW |
| E. 欠定澄清探针 | 16 | 搜索低效/擅自假设（~23%） | "哪个账户推荐奖励最高"——不说账户类型；KB 里 checking 才是最高 |
| F. Grounding / 拒答 | 2 | — | 信息在 KB 中根本不存在，正确行为是承认找不到 |
| G. 资格拒绝 | 9 | — | 用户请求不合规，agent 必须拒绝并解释 |
| H. 中途状态变更 | 8 | — | flow rule 在第 N 轮改写 DB（"钱包在夹克里找到了"） |

工具调用数目标分布：均值 9.52、min 1、max 33。建议按 `round(LogNormal(μ=2.05, σ=0.72))` 采样后裁剪到 [1,33]，并把 F 类（拒答）强制设为 1–2，把 C 类设在 12–20，个别 A/C 类手工拉到 30+。

### F3. flow-based 用户模拟器

论文强调 user simulator 不能变成"unwitting oracle"——不能提前泄漏未来状态。生成 user prompt 时的硬规则：

1. prompt 里**只写用户自己知道的信息**（人设、目标、验证信息、偏好），不写 DB 内部状态、不写正确工具名、不写正确顺序
2. 条件规则一律写成 `如果 agent 做了 X，你就说 Y`，且 X 必须是**用户可观测**的（agent 的自然语言话术或已告知用户的结果），不能是"如果 agent 调用了 tool_1234"这类内部事件——除非该调用结果确实会被告知用户
3. 约束**逐步透露**而非一次性倒出（Sample 1 的写法），制造欠定 → 澄清的压力
4. 加一条通用 fallback：`If given multiple options: "I really don't want to compare options. Which ONE ... ?"`

**模拟器可靠性抽检**：随机抽 2 条轨迹/任务，标注每条 user utterance 为 error-free / task-benign / task-critical，task-critical 率必须 ≤3%（官方 4/194 ≈ 2%）。超标则回去改 flow rule 的触发条件。

### F4. 种子数据库

表结构照官方观感：`users`, `bank_accounts`, `credit_card_accounts`, `debit_cards`, `transaction_history`, `credit_card_transaction_history`, `referrals`, `cash_back_disputes`, `closure_reason_history`, `replacement_orders`, `credit_card_account_flags`, `verification_records`。

ID 方案要**刻意保持异构**（官方就是异构的，这本身是分布特征）：

| 实体 | 格式 | 例 |
|---|---|---|
| user_id | 姓名首字母(2) + hex(8)，少数纯 hex(10) | `jc61f7a8d2`, `6680a37184` |
| bank_account_id | hex(16) 或语义前缀 | `61c9d8e7f6a5b432`, `chk_lj82d4f1a9` |
| credit_card_account_id | `cc_{user_id}_{class_abbr}` | `cc_224959b99e_plat` |
| debit_card_id | `dbc_{user_id}_{acct}` 或 `dbc_{hex10}` | `dbc_lj82d4f1a9_bluest`, `dbc_538bfb9cba` |
| dispute_id | `dsp_{base36(8)}` | `dsp_7a3p...` |

**背景噪声**：除 97 个任务用户外，再合成 200–400 个不参与任何任务的用户及其账户/交易，防止 agent 通过"数据库里只有一个符合条件的东西"走捷径。

---

## 8. Stage G：可解性与 gold 最小性证明

论文 Stage 5 由两位未参与出题的评审人工审。自动化对应物：

### G1. 可解性（对应"expected final database state was correct"）

把 DB + 51+14 个工具实现成可执行的 Python 环境（带 precondition 断言）。从 `initial_db_patch` 出发，按 `expected_actions` 逐步执行：

```
assert 所有 precondition 满足        # 顺序正确性
assert final_state ≡ target_final_db_state  （按 task 声明的比较键）
```

任何 precondition 触发说明 `expected_actions` 的顺序错了 —— 这条自动检查能抓出绝大多数出题错误。

### G2. gold 集完整性 & 最小性

把每篇文档映射到它**揭示的知识元素**集合：`reveal(D) = {变量 id} ∪ {工具签名 id} ∪ {policy 规则 id}`。
任务需要的知识元素集 `Req(t)` 由 `expected_actions` + `constraints` 反推。

- **完整性**：`Req(t) ⊆ ⋃_{D ∈ gold(t)} reveal(D)`
- **最小性**：`∀ D ∈ gold(t): Req(t) ⊄ ⋃_{D' ∈ gold(t)\{D}} reveal(D')`

即 gold 集是 `Req(t)` 的一个**极小覆盖**。这比人工"逐篇删掉试试"快几个数量级，且可在任务数增长到数百时保持成立。

### G3. 事后重审（对应"re-audited to ensure that no unintended shortcuts"）

跑完一轮大规模实验后，对**所有成功轨迹**做自动扫描：

- 成功但 `document_recall < 40%` → 疑似猜中或走捷径，标记复查
- 成功但 `expected_actions` 未被完整覆盖 → 目标状态比较键太松，需收紧
- 成功但从未调用 `get_current_time()` 却答对了时效题 → 促销陷阱失效

---

## 9. Stage H：同分布验收门

三层，全过才发布。

**L1 文本统计**
- token 长度分布 KS 检验 vs 官方，p > 0.05
- 文档原型比例 χ² 检验，p > 0.05
- MTLD、平均句长、表格密度、bullet 密度 落在官方 ±10%

**L2 语义/检索分布**
- 用 `text-embedding-3-large` 编码两个语料，比较质心余弦 > 0.85，MMD 落在自助法置信区间内
- **检索难度代理**（比 L1 重要）：同一 agent + 同一检索器下，gold 文档 recall@10 与官方同档 —— Opus+dense ≈ 57%，GPT-5.2(none)+dense ≈ 28%

**L3 行为分布（最终裁决）**
用一个固定参照 agent（建议 Claude-4.5-Sonnet high + BM25，跑得快且官方基线明确 = 16.75）跑全量任务：

| 检查 | 官方 | 容差 |
|---|---|---|
| 参照配置 pass^1 | 16.75 | ±3.0 |
| gold 配置 pass^1 | 33.76 | ±4.0 |
| gold − 检索 差值 | ≈ −17 | ±5 |
| pass^1 → pass^4 衰减比 | 22.42 → 10.31 ≈ 0.46 | ±0.12 |
| no-knowledge pass^1 | ≈2% | ≤4% |
| long-context pass^1 | ≈12% | ±5 |

**no-knowledge ≈ 2% 和 long-context ≈ 12% 这两条最关键**：前者证明任务确实依赖检索到的知识（而非常识可推），后者证明非 gold 文档构成了真实噪声而非无关填充。任何一条大幅偏高，说明合成语料"太干净"或"任务太好猜"，必须回到 Stage A 加强干扰项。

---

## 10. Stage I：增量刷新回路

对应论文 Stage 4 —— "selectively re-run the structured-to-unstructured generation pipeline for affected portions"。

维护三张索引：`var → docs`、`var → tasks`、`tool → docs`。变量值变更时：

```
1. 受影响文档 = var→docs[v]
2. 若只是值变化且类型/量纲不变 → 只跑 Pass 2 确定性填充（不调 LLM，秒级）
3. 若变量增删或语义变化 → 重跑该 feature 的 Stage C+D
4. 重跑 L1–L9 中受影响的检查
5. 受影响任务 = var→tasks[v] → 重跑 G1+G2
6. 全局 L7（长度均值）与 H 层验收在批次末统一跑一次
```

每篇文档存内容哈希 `sha256(title || sorted(vars) || model || persona || seed)`，哈希不变则复用缓存。这让"改一个 APY 数值"的代价从整库重生成降到几十毫秒。

---

## 11. 目录结构与实施顺序

```
tau-banking-synth/
├── schema/          categories.yaml  features.yaml  variables.yaml  interactions.yaml
├── registry/        tools.yaml
├── plan/            allocation.json  titles.json
├── render/          {doc_id}.md.tmpl        # 含 [[var]]
├── kb/              {doc_id}.md  INDEX.md   # 最终语料（terminal 配置直接挂载）
├── db/              base_seed.json  patches/{task_id}.json
├── tasks/           {task_id}.yaml
├── env/             tools_impl.py  db.py  preconditions.py
├── lint/            L1..L9.py
├── verify/          solvability.py  minimality.py  audit_trajectories.py
├── accept/          text_stats.py  retrieval_stats.py  behavior_gate.py
└── run.py           # 编排 + 内容哈希缓存
```

建议实施顺序（每步都能独立验收）：

1. **先搭 env/**（DB + 工具 + precondition）。没有可执行环境，G1/G2 无从谈起，任务质量会失控。
2. **手写 5 个任务 + 对应最小 KB**（每类原型各挑一个），跑通 G1/G2/L4/L5。这一步会暴露 80% 的 schema 设计问题。
3. **补齐 Stage A 的 21 类别 / 71 主题骨架**，跑 CSP 赋值。
4. **跑 Stage C+D 全量渲染 698 篇**，过 L1–L9。
5. **扩到 97 个任务**，全量跑 G。
6. **跑 H 层验收**，按偏差回调：pass^1 偏高 → 加干扰项 / 提高碎片化 φ；no-knowledge 偏高 → 检查是否有常识可推的任务。
7. **跑一轮大规模实验后做 G3 事后重审**。

---

## 12. 风险与常见坑

| 风险 | 症状 | 对策 |
|---|---|---|
| **LLM 幻觉数值** | 文档里的费率和 DB 对不上，任务变成不可解 | Pass 1 强制占位符 + L1 数值溯源门禁（最重要的一条） |
| **语料太"干净"** | no-knowledge / long-context pass^1 明显高于官方 | 增加近似干扰项、同色系跨类别复用、促销时效陷阱 |
| **风格聚簇** | dense 检索 recall 异常高，任务变简单 | 强制 (model, persona, register) 在 feature 内混合；检查 embedding 空间是否按 model 可分 |
| **gold 集虚胖** | 平均 gold 数远超 18.6，任务实际不可解 | G2 极小覆盖检验；调低碎片化 φ |
| **user simulator 当预言机** | agent 未推理却被用户"提示"到正确路径 | F3 规则 1–2；抽检 task-critical 率 ≤3% |
| **意外跨 feature 碰撞** | 两个任务的唯一解互相冲突 | A3 独立性原则 + L4 在**全量任务集**上跑（不是逐任务跑） |
| **后缀被记忆** | 模型在没检索到文档时也猜对工具名 | 每次发布重掷 4 位后缀；no-knowledge 基线里检查工具名命中率 |
| **上下文溢出掩盖失败** | 长任务因截断失败，被误判为推理能力问题 | 复刻论文的截断策略（淘汰最旧检索输出的 1/4 + 占位提示），并记录截断率（官方 1–3%，仅 GPT-5.2 high） |

---

## 附：与官方命名的对应

论文中该 benchmark 名为 **τ-Knowledge**，其中唯一的 domain 是 **τ-Banking**（构建在 τ-Bench / τ²-bench 的对话环境之上）。如果你手上的代码库把 domain 目录叫 `tau3-banking` 或类似名字，注意区分：`τ²-bench` 是 dual-control 环境（Barres et al., 2025），而本文的 τ-Banking 的核心新增是**可发现工具 + 700 篇非结构化 KB**。合成 pipeline 的产物应当直接落进该 domain 的 `documents/` 与 `db.json`（terminal 配置额外需要 `INDEX.md`）。
