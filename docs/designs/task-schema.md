
> **任务骨架（TaskSkeleton）与 KB 变体（KBVariant）解耦。** 骨架里只存符号引用（`product:sky_blue#apy_rate`），不存字面值；具体任务 = 骨架 × 变体，在 emit 时绑定。这样 8k 骨架 × 12 变体 ≈ 96k 具体任务，而且同一个骨架在不同变体下的正确答案不同——模型没法背，只能学会"去查"。

```
Layer 0  WorldSpec        从 698 篇文档反解出的结构化世界
Layer 1  ConstraintSolver 产品属性空间上的约束求解 + 唯一性判定
Layer 2  TemplateDSL      任务族模板（带难度轴的生成器）
Layer 3  TaskSkeleton     符号化任务骨架
Layer 4  Binder + Gates   绑定变体 → 具体任务 → 七道质检闸门
```

---

## Layer 0：WorldSpec

这是整个引擎的唯一真源。所有下游产物都必须能追溯到它的某个字段。

```python
@dataclass
class AttributeValue:
    value: Any
    type: Literal["money","percent","days","count","bool","enum","date","absent"]
    doc_refs: list[DocId]          # 该属性在哪些文档里被提及（gold_docs 的来源）
    surface_forms: list[str]       # 文档里的表述变体，供 BM25 query 生成器用
    variant_policy: Literal["resample","fixed","scale"]   # KB 变体化时怎么扰动

@dataclass
class Product:
    product_id: str                 # "sky_blue_business_checking"
    category: str                   # 21 个品类之一
    display_name: str
    attributes: dict[str, AttributeValue]
    eligibility: list[Predicate]    # 开户前置条件（账户年龄、无 CLOSED 状态等）
    linked_effects: list[LinkedEffect]   # ★ 跨产品耦合，14.5% 失败桶的源头
```

`LinkedEffect` 是最值得投入的字段，它编码"绑定 Crypto Cash Back 卡可以给储蓄账户加 APY，但加完仍不如另一个基础利率更高的账户"这类陷阱：

```python
@dataclass
class LinkedEffect:
    condition: Predicate            # "user.has_product(crypto_cashback_card)"
    target_attr: str                # "apy_rate"
    op: Literal["add","mul","set","waive"]
    magnitude: float
    doc_refs: list[DocId]           # 关键：这条规则单独躺在另一篇文档里
```

工具规格：

```python
@dataclass
class ToolSpec:
    tool_name: str                  # "update_transaction_rewards_3847"
    kind: Literal["permanent_agent","discoverable_agent","discoverable_user"]
    params: list[ParamSpec]         # name / type / required / enum_values
    doc_refs: list[DocId]           # 签名出现在哪篇文档（不检索到就无法解锁）
    preconditions: list[Predicate]  # 调用前 DB 必须满足的条件
    effects: list[DBEffect]         # 结构化的写操作，用于推演 gold 终态
    unlock_required: bool
    is_trap: bool = False           # 名字相似但语义错误的陷阱工具
    confusable_with: list[str]      # 易混工具，用于生成负样本
```

流程文档编译成 DAG：

```python
@dataclass
class PolicyDAG:
    policy_id: str                  # "cc_retention_protocol"
    doc_refs: list[DocId]
    nodes: dict[StepId, PolicyStep] # kind: check | log | offer | mutate | ask_user
    edges: list[tuple[StepId,StepId]]
    hard_order: bool                # True = 乱序即判失败
    blocking_rules: list[BlockRule] # "有未决争议 → 阻断关户"
```

23 张 TransactionalDB 表的 schema 也在这一层，用于 `DBEffect` 的推演和终态断言。

---

## Layer 1：约束求解器

每个品类只有 7~8 个产品，暴力枚举即可，不需要 SAT solver。

```python
class ConstraintSet:
    hard: list[Constraint]     # op: ge|le|eq|in|absent, attr, value
    objective: Optional[Objective]   # maximize/minimize + attr
    tie_break: list[TieBreak]        # promotion | linked_effect | tenure

def solve(cs, category, world, sim_date) -> list[Product]:
    cands = [p for p in world.products_of(category) if all(c.holds(p) for c in cs.hard)]
    cands = apply_linked_effects(cands, world, cs.context)     # 先算耦合后的有效值
    if cs.objective: cands = argopt(cands, cs.objective)
    for tb in cs.tie_break: cands = tb.apply(cands, sim_date)  # 促销要判有效期
    return cands
```

**唯一性判定就是任务是否 well-posed 的定义**：`len(solve(...)) == 1`。这是生成器唯一的正确性保证来源，也是论文里"约束可以直接对结构化库做单元测试式验证"那句话的落地。

生成约束集的方向是**反向的**：先随机选定目标产品 `p*`，再合成一组约束使 `p*` 成为唯一解。

```python
def synth_constraints(p_star, category, world, n_constraints, n_distractors):
    others = world.products_of(category) - {p_star}
    cs = []
    # 每条约束负责淘汰至少一个竞争者，且 p* 必须满足
    while others and len(cs) < n_constraints:
        attr, op, val = sample_separating_constraint(p_star, others)
        cs.append(Constraint(attr, op, val))
        others = {p for p in others if Constraint(attr,op,val).holds(p)}
    # 注入 near-miss 干扰：恰好违反 1 条约束的产品必须留在候选池里
    ensure_near_misses(cs, world, k=n_distractors)
    assert solve(cs, category, world) == [p_star]
    return cs
```

`sample_separating_constraint` 要**优先选那些属性分散在不同文档里的**（查 `doc_refs`），这样才能把 gold_docs 数推到 τ³ 真实的 18.6 篇量级，逼出多跳检索。

---

## Layer 2：模板 DSL

一个模板 = 采样器 + 难度轴 + 发射器。按第一轮列的失败归因分配权重：

| template_id | 权重 | 对治 |
|---|---|---|
| `multi_constraint_selection` | 18% | 多跳比价（14.5% 桶） |
| `underspecified_clarification` | 16% | 凭假设行动（23% 桶） |
| `procedural_execution` | 15% | 流程 DAG 严格执行 |
| `tool_discovery_chain` | 14% | unlock 协议（社区实测首要失败） |
| `topological_ordering` | 10% | 隐式顺序（5% 桶） |
| `state_assertion_conflict` | 8% | 过度采信用户（4% 桶） |
| `temporal_validity` | 6% | 促销有效期 / 账龄 |
| `user_delegated_action` | 5% | 4 个 user-side 工具 |
| `mid_conversation_state_change` | 4% | flow rule 中途改状态 |
| `grounding_refusal` | 2% | KB 里没有 → 明确说不知道 |
| `trap_tool_avoidance` | 2% | 陷阱工具 |

模板声明式定义：

```yaml
template_id: topological_ordering
family: sequencing
difficulty_axes:
  n_operations:      {range: [3, 5]}
  n_hidden_deps:     {range: [1, 3]}       # 阻断边的数量
  dep_source_spread: {range: [1, 3]}       # 依赖规则分散在几篇文档
  user_pushback:     {choices: [none, once, twice]}   # 用户质疑改顺序
sampler: |
  ops = sample_operations(world, n_operations)          # 开户/关户/升级
  deps = inject_blocking_edges(ops, world, n_hidden_deps)
  assert has_cycle(user_order(ops)) or violates(deps, user_order(ops))
  gold_order = topo_sort(ops, deps)
  assert is_feasible(gold_order, world)
emits:
  gold_docs: docs_of(deps) ∪ docs_of(ops) ∪ docs_of(eligibility)
  forbidden_prefix: [ops in user_order]   # 按用户顺序执行前两步 → 立即判失败
```

难度轴的作用有两个：课程学习的采样维度，以及事后做 pass-rate 归因回归（哪条轴最卡模型）。

---

## Layer 3：TaskSkeleton（符号化）

这是落盘的核心资产。**注意所有值都是符号引用，不是字面量。**

```jsonc
{
  "skeleton_id": "sk_topo_00417",
  "template_id": "topological_ordering",
  "difficulty": {"n_operations": 4, "n_hidden_deps": 2, "dep_source_spread": 3,
                 "user_pushback": "once", "n_gold_docs": 14, "n_tool_calls": 12},

  "initial_state": {
    "user_ref": "$persona.user_id",
    "db_patch": [
      {"table": "accounts", "op": "insert",
       "row": {"account_id": "$ref:acct_A", "type": "personal_savings",
               "class": "@product:bronze_saver", "opened_days_ago": 12}},
      {"table": "accounts", "op": "insert",
       "row": {"account_id": "$ref:acct_B", "type": "personal_checking",
               "class": "@product:evergreen_checking", "opened_days_ago": 900}}
    ],
    "agent_config": {"auto_resolve_disputes": false}
  },

  "user_scenario": {
    "persona": {"name": "$gen:name", "age": "$gen:age", "city": "$gen:city",
                "occupation": "$gen:occupation"},
    "verification_info": {"name": "$persona.name", "dob": "$gen:dob",
                          "email": "$gen:email", "address": "$gen:address"},
    "goal": "完成一次账户重组：关掉两个账户，开两个新账户",
    "opening_template": "topo_four_requests_v3",
    "requested_order": ["close:$ref:acct_A", "open:business_checking",
                        "close:$ref:acct_B", "open:personal_savings"],
    "constraints_revealed": [
      {"reveal_when": "asked_about:business_checking",
       "constraint": {"attr": "@attr:monthly_maintenance_fee", "op": "eq", "value": 0}},
      {"reveal_when": "asked_about:business_checking",
       "constraint": {"attr": "@attr:minimum_balance", "op": "absent"}},
      {"reveal_when": "asked_about:personal_savings",
       "constraint": {"attr": "@attr:monthly_withdrawal_limit", "op": "ge", "value": 15}}
    ],
    "flow_rules": [
      {"rule_id": "r1",
       "trigger": {"type": "agent_proposes", "match": {"intent": "reorder_operations"}},
       "response_template": "pushback_why_not_closures_first",
       "once": true},
      {"rule_id": "r2",
       "trigger": {"type": "agent_explains", "match": {"intent": "dependency_explanation"}},
       "response_template": "concede_expert",
       "once": true},
      {"rule_id": "r3",
       "trigger": {"type": "agent_asks", "match": {"slot": "savings_balance"}},
       "response_template": "reveal_balance", "binds": {"amount": "$gen:balance_2500_3000"}}
    ],
    "known_unknowns": ["account_opening_dates", "tenure_requirements",
                       "closed_status_blocking_rule"]
  },

  "hidden_dependencies": [
    {"dep_id": "d1", "rule": "@policy:business_checking_eligibility#no_closed_accounts",
     "blocks": "open:business_checking", "if_done_first": "close:$ref:acct_A",
     "doc_refs": ["@doc:biz_checking_eligibility"]},
    {"dep_id": "d2", "rule": "@policy:savings_open#min_tenure_14d",
     "blocks": "open:personal_savings", "if_done_first": "close:$ref:acct_B",
     "doc_refs": ["@doc:savings_tenure_policy"]}
  ],

  "solution": {
    "gold_order": ["open:business_checking", "close:$ref:acct_A",
                   "open:personal_savings", "close:$ref:acct_B"],
    "gold_actions": [
      {"tool": "log_verification", "args": {"name": "$persona.name",
                                            "user_id": "$persona.user_id"}},
      {"tool": "unlock_discoverable_agent_tool",
       "args": {"agent_tool_name": "@tool:get_all_accounts"}},
      {"tool": "call_discoverable_agent_tool",
       "args": {"agent_tool_name": "@tool:get_all_accounts",
                "arguments": {"user_id": "$persona.user_id"}}},
      {"tool": "unlock_discoverable_agent_tool",
       "args": {"agent_tool_name": "@tool:open_bank_account"}},
      {"tool": "call_discoverable_agent_tool",
       "args": {"agent_tool_name": "@tool:open_bank_account",
                "arguments": {"user_id": "$persona.user_id",
                              "account_type": "business_checking",
                              "account_class": "@solve:biz_checking_solution"}}}
    ],
    "gold_docs": ["@doc:biz_checking_eligibility", "@doc:savings_tenure_policy",
                  "@doc:tool_open_bank_account", "@doc:tool_close_account", "..."]
  },

  "evaluation": {
    "db_assertions": [
      {"table": "accounts", "match": {"type": "business_checking"},
       "expect": {"class": "@solve:biz_checking_solution", "status": "ACTIVE"},
       "cardinality": 1},
      {"table": "accounts", "match": {"account_id": "$ref:acct_A"},
       "expect": {"status": "CLOSED"}},
      {"table": "accounts", "match": {"type": "personal_savings",
                                      "account_id": {"$ne": "$ref:acct_A"}},
       "expect": {"class": "@solve:savings_solution", "status": "ACTIVE"},
       "cardinality": 1}
    ],
    "no_extra_writes": {"tables": ["accounts","credit_card_accounts","disputes"],
                        "allow": "$solution.db_delta"},
    "action_checks": {
      "required_ordering": [["open:business_checking", "close:$ref:acct_A"],
                            ["open:personal_savings", "close:$ref:acct_B"]],
      "forbidden": [{"tool": "@tool:close_account", "before": "open:business_checking"}]
    },
    "nl_assertions": ["向用户解释了为什么必须调整操作顺序"],
    "reward_basis": ["db_assertions", "no_extra_writes"]
  }
}
```

三个容易被忽略但很关键的字段：

**`no_extra_writes`** — 不加这个，RL 会学出"把所有可能的操作都做一遍"的散弹枪策略，DB 断言照样通过。必须显式声明白名单之外的写操作即判失败。这是最典型的 reward hacking 入口。

**`known_unknowns`** — 用户模拟器绝对不能说出的信息。τ³ 论文专门批评过"user simulator 无意中充当 oracle"的问题；不显式约束的话，蒸馏出来的 user sim 会在训练中泄露答案，RL 学出的策略上真榜直接崩。

**`constraints_revealed[].reveal_when`** — 欠定程度的控制阀。`reveal_when: always` 是简单模式，`asked_about:X` 是必须主动问才给，这条轴直接对应 23% 的失败桶。

---

## Layer 4：Binder + 七道闸门

### Binder

```python
def bind(skeleton: TaskSkeleton, variant: KBVariant, seed: int) -> ConcreteTask:
    world = variant.world_spec
    ctx = {}
    ctx |= resolve_gen_fields(skeleton, seed)          # $gen:* → 具体人名/日期
    ctx |= resolve_refs(skeleton, seed)                # $ref:* → 账户 ID
    ctx |= {k: solve(cs, ...) for k, cs in skeleton.solve_targets.items()}  # @solve:*
    ctx |= {r: world.deref(r) for r in skeleton.symbol_refs()}              # @product/@tool/@doc
    task = substitute(skeleton, ctx)
    task.solution.gold_actions = reproject_tool_names(task, world)  # ★ 变体里工具后缀不同
    return task
```

`reproject_tool_names` 是变体化的核心：变体 v7 里 `open_bank_account` 的后缀可能是 `_9134` 而不是 `_4821`。模型只能通过检索文档拿到正确后缀，没有第二条路。

### 七道闸门（按成本从低到高，早失败早丢弃）

| # | 闸门 | 判据 | 处理 |
|---|---|---|---|
| 1 | **唯一性** | `len(solve(cs)) == 1` | 生成期内联，失败重采样 |
| 2 | **确定性** | 同 seed 两次 bind 的 DB delta 逐字节相同 | 失败=生成器有 bug，报警 |
| 3 | **可解性** | 脚本化 oracle agent（喂 gold_docs + gold_actions）在真 env 里执行，db_assertions 全过 | 失败即丢弃 |
| 4 | **必要性（反捷径）** | 关掉 KB 访问跑 k=4，全失败才算合格 | 通过=退化任务，丢弃 |
| 5 | **最小性** | 逐一移除 gold_docs 中的文档，oracle 必须失败 | 抽样 10% 跑；冗余文档从 gold_docs 剔除 |
| 6 | **flow 可达性** | 用真 user sim 跑 3 条轨迹，检查每条 flow_rule 至少触发一次 | 未触发的 rule 说明 trigger 写错了 |
| 7 | **区分度** | 当前策略 k=8 采样，`0.05 < pass_rate < 0.95` | 区间外的进冷/热池，不进主训练集 |

闸门 4 是最重要的一道。论文里 no-knowledge 配置下平均 pass^1 只有 ~2%，那个 2% 就是两道接地检查题——你的合成集必须复现这个性质，否则模型会学会"不查也能蒙对"，这个习惯带上真榜是灾难。

闸门 7 要**在线重跑**：模型在变强，昨天区分度好的任务今天可能全对了。建议每 200 个 RL step 对采样池重新打一次 pass_rate 标签。

---

## 规模与配比

```
骨架:      8,000  （按上表族权重分配）
KB 变体:      12  （1 个原始 + 11 个扰动）
具体任务: ~96,000
```

划分：

- **训练池** 85%，按 **KB 子图** 切分，不是随机切。同一条 policy DAG 涉及的骨架必须整体归入同一侧，否则会有策略级泄漏。
- **合成留出** 10%，用来做泛化诊断（合成留出 vs 真 97 题的分数相关性应 >0.8，这是你的生成器质量指标）。
- **困难池** 5%，pass_rate < 0.05 的，单独做拒绝采样 SFT。

难度课程建议按 `n_gold_docs` 和 `n_tool_calls` 两轴推进：
`(1~3 docs, ≤5 calls)` → `(4~8, 6~12)` → `(9~14, 13~22)` → `(15~20, 23~33)`。最后一档要覆盖到 τ³ 的上限（18.6 篇 / 33 次），否则模型在真榜的长尾难题上会集体截断。

---

## 两个实现上的坑

**flow_rule 的 trigger 匹配不能用正则。** `agent_proposes: reorder_operations` 这种意图判定必须用一个小分类器（或 user sim 自己判）。用关键词匹配的话，模型会学会说特定的话来骗触发器——这是另一个 reward hacking 入口，而且很隐蔽，因为它污染的是 user sim 而不是奖励函数。

**`@solve:*` 必须在 bind 时求解，不能预存。** 变体里数值重采样后，同一组约束的唯一解可能变成另一个产品，甚至变成无解。Binder 里必须重跑闸门 1，无解就换 seed 重 bind，不能沿用骨架里缓存的答案。

---

要不要我接着把 **user simulator 的蒸馏与校准流程**（怎么保证它和 GPT-5.4 Mini 的行为分布对齐、怎么量化"泄露答案"的程度）展开？那是这套东西里第二个容易翻车的地方，而且和这里的 `known_unknowns` / `flow_rules` 直接咬合。