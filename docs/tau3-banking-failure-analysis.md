# tau3-banking 失败模式分析（run: `tau3_banking_dots3_max`）

## 0. 数据来源与口径

| 项 | 值 |
|---|---|
| 结果目录 | `data/simulations/tau3_banking_dots3_max/` |
| 被测 agent | `openai/d3note`（`reasoning_effort: max`, `temperature 0.7`, `top_p 0.95`） |
| user simulator | `openai/gpt-5.4-mini`（`temperature 0`, `reasoning_effort: medium`） |
| domain | `banking_knowledge` |
| 规模 | 97 个任务 × 5 trial = 485 条 simulation |
| 预算 | `max_steps 200`, `max_errors 10` |
| git commit | `363133ada1936491fb5bcec33cd62c3518a99f65` |

奖励口径：431 条以 `DB`（终态数据库比对）判分，45 条以 `ACTION` 判分，5 条为 `DB + NL_ASSERTION`，4 条因触顶无 breakdown。**奖励是 0/1 的**，没有部分分——任何一个必需动作缺失、参数错误，或多做了一次写操作，整条轨迹都归零。

## 1. 总体表现

```
pass@1 (5 trial 平均)     20.2%     (98/485)
pass@5 (至少一次通过)      41.2%     (40/97 任务)
pass^5 (五次全过)           4.1%     (4/97 任务)
不稳定任务 (0 < 通过率 < 1) 36/97
全灭任务 (五次全失败)       57/97
```

终止原因几乎全是 `user_stop`（481），只有 4 条撞 `max_steps`。**也就是说模型极少"卡死"，它几乎总是自信地走完流程并给出一个错误的终态。** 这是本次评测最重要的整体观察：失败不是能力崩溃，而是**静默的错误完成**。

硬错误同样罕见：485 条轨迹里工具调用报错总共只有 9 次（6 条轨迹）。失败全部是语义层面的。

### 失败轨迹更长、更贵

| | 通过 | 失败 |
|---|---|---|
| 平均消息数 | 42.4 | 64.6 |
| 平均工具调用数 | 17.3 | 27.8 |
| 平均 `KB_search` 次数 | 8.8 | 12.3 |
| 平均 `grep` 次数 | 0.85 | 2.60 |
| `KB_search` ≥ 20 次的轨迹占比 | 9.2% | 18.6% |

失败轨迹的检索量是通过轨迹的 1.4–3 倍。**更多的检索不是在补足信息，而是在打转**（见 §3）。

### 通过率随任务动作数急剧衰减

| 期望动作数 | 任务数 | 平均 pass@1 |
|---|---|---|
| 1–4 | 30 | 0.40 |
| 5–9 | 29 | 0.19 |
| 10–14 | 12 | 0.15 |
| 15–19 | 13 | 0.02 |
| 20–24 | 7 | 0.03 |
| 25–29 | 6 | 0.00 |

超过 15 个动作的 26 个任务里，5 次 trial 共 130 次机会只通过了 3 次。在 0/1 判分下，单步正确率 p 对应整体通过率约 p^n——这条曲线反推出的单步正确率大约在 0.85–0.90 之间，说明**问题不是某类任务不会做，而是长程无损执行能力不足**。

### 按业务族拆分

| 业务族 | pass@1 |
|---|---|
| 争议/退单 dispute (189 条) | 0.12 |
| 账户开立/关闭 (75 条) | 0.09 |
| 借记卡操作 (30 条) | 0.13 |
| 推荐 referral (35 条) | 0.23 |
| 账户补款 statement/account credit (97 条) | 0.35 |
| 其他 (59 条) | 0.39 |

## 2. 失败模式总览

把 485 条轨迹的 `action_checks` 全部展开（共 2214 个未匹配的期望动作），按根因归类：

| # | 失败模式 | 规模 | 判定依据 |
|---|---|---|---|
| **F1** | 所需的 discoverable 工具**从未被调用** | 1470 个动作 / 173 条失败轨迹 | 期望动作的工具名在整条轨迹中不出现 |
| **F2** | 工具调用了但**参数值错误** | 371 处 | 共有 key 上取值不同 |
| **F3** | **枚举不全**：同一工具该调 N 次只调了 M 次 | 239 处 | 参数格式正确、调用次数不足 |
| **F4** | **过度动作**：做了期望之外的写操作，污染终态 DB | 11 条轨迹"动作全对仍 0 分" | `action_match` 全 True 但 `db_match` False |
| **F5** | **过早升级**转人工 | 27 条无端转人工 / 58 条失败轨迹含转人工 | 期望动作里没有 transfer 却调用了 |
| **F6** | 转人工时 **`reason` 枚举值选错** | 36/65 次 transfer 参数不匹配 | 见 §2.6 |
| **F7** | **凭空捏造参数字段** | 47 处 | 期望参数的超集 |
| **F8** | **完全不作为** | 20 条轨迹 | 任务需要写操作，agent 一个写操作都没做 |

下面逐条展开。

### F1 — 知识库里的工具没被找到（最大头）

这是压倒性的第一失败原因。173/387 条失败轨迹里，至少有一个必需工具**根本没被调用过**。缺失最多的：

```
165  get_bank_account_transactions_9173
154  get_all_user_accounts_by_user_id_3847
127  get_user_dispute_history_7291
124  get_debit_cards_by_account_id_7823
110  file_credit_card_transaction_dispute_4829
 92  open_bank_account_4821
 89  get_pending_replacement_orders_5765
 63  close_debit_card_4721
 61  order_debit_card_5739
```

注意缺失榜首几乎全是**只读的取数工具**。domain policy 规定 discoverable 工具必须先 `unlock_discoverable_agent_tool` 再 `call_discoverable_agent_tool`，且工具名只能来自知识库。模型的典型行为是：**跳过"先把用户全部账户/全部交易拉出来"这一步，直接凭对话里用户口述的信息去做写操作**。于是后续所有 ID、金额、账户类型都建立在用户的模糊描述而非数据库事实上。

同时 `get_credit_card_accounts_by_user`（173 次）和 `get_credit_card_transactions_by_user`（100 次）是被**多调**最多的工具——模型偏好它已经熟悉的常规工具，而不是去知识库里发现场景专用的 discoverable 工具。这是一种**工具发现的惰性替代**。

### F2 — 参数值错误：集中在"需要按政策推算的字段"

371 处取值错误里，错得最多的字段有非常清晰的共性——**它们都不是抄来的，而是要根据知识库规则算出来的**：

| 次数 | 字段 | 典型错误 |
|---|---|---|
| 57 | `submit_cash_back_dispute_0589.transaction_id` | 选错了该争议的交易 |
| 47 | `file_debit_card_transaction_dispute_6281.customer_max_liability_amount` | **期望 50，填 500** |
| 37 | `open_bank_account_4821.account_class` | 选错账户档位 |
| 31 | `apply_checking_account_credit_5829.amount` | 补款金额算错 |
| 24 | `file_credit_card_transaction_dispute_4829.eligible_for_provisional_credit` | **期望 false，填 true** |
| 20 | `submit_interest_discrepancy_report_7294.amount_difference` | 利息差额算错 |
| 19 | `submit_interest_discrepancy_report_7294.expected_apy` | **期望 6.85，填 6.875** |
| 16 | `file_credit_card_transaction_dispute_4829.card_action` | **期望 `cancel_and_reissue`，填 `keep_active`** |
| 16 | `update_transaction_rewards_3847.new_rewards_earned` | 返现重算错 |

具体高频错对：

```
28×  customer_max_liability_amount:  期望 50      → 填 500
24×  eligible_for_provisional_credit: 期望 false   → 填 true
16×  card_action:                    期望 cancel_and_reissue → 填 keep_active
11×  order_debit_card.card_design:   期望 PREMIUM  → 填 CLASSIC
 9×  dispute_category:  期望 card_present_fraud    → 填 card_not_present_fraud
 8×  transaction_type:  期望 pin_purchase          → 填 online_purchase
 6×  expected_apy:      期望 6.85                  → 填 6.875
 6×  delivery_option:   期望 EXPEDITED             → 填 STANDARD
```

三个可分离的子模式：

1. **分级规则（tier）套错档**。Reg-E 类的责任上限 50 → 500 是最典型的：卡还在手里、及时上报应落在 \$50 档，模型套到了 \$500 档。APY 也一样（6.85 vs 6.875、3.25 vs 3.0、4.275 vs 4.25），是**没找全叠加的 bonus 项**或**把 base 当成了最终值**。
2. **交易性质分类错**：`pin_purchase` 被判成 `online_purchase`/`signature_purchase`，`card_present_fraud` 被判成 `card_not_present_fraud`。这直接连锁导致责任上限和临时信用资格算错——**一个分类错误会同时打错 3–4 个字段**。
3. **系统性保守偏置**。`eligible_for_provisional_credit` 错误里 24 次是 false→true（过度给付），`card_action` 16 次是 cancel_and_reissue→keep_active（该换卡不换），`card_design`/`delivery_option` 全部错向默认档 CLASSIC/STANDARD，`expedited_shipping` 6 次该 true 填 false。**模型倾向于选"默认/温和/不打扰用户"的那个值，而任务的正确答案往往是政策规定的特殊档位。**

### F3 — 枚举不全：找不齐所有符合条件的对象

239 处"参数完全正确但次数不够"。最典型的是返现争议类任务：用户有 6 笔返现计算错误的交易，期望对每一笔各调一次 `submit_cash_back_dispute_0589`，模型只找出了 1–2 笔。

```
task_029  期望 6 笔 txn 各一次     实际只提交 txn_adea68821a1d 一笔
task_027  期望 4 笔               实际提交 2 笔，且其中 0 笔命中期望集合
task_019  期望 4 笔               实际 2 笔命中
```

被"少调"最多的工具：`get_all_user_accounts_by_user_id_3847`(55)、`open_bank_account_4821`(33)、`get_bank_account_transactions_9173`(25)、`get_debit_cards_by_account_id_7823`(24)。

这与 F1 是同一个病根的两面：**模型没有先把对象全集拉出来，就开始逐个处理**，于是处理了对话中被提到的那一两个，漏掉了数据库里其余符合条件的。注意 `open_bank_account_4821` 同时出现在"少调 33"和"多调 35"两张榜上——模型对"该开几个户、开哪几种"的判断是随机游走式的。

### F3b — 用户工具的交接：讲错了参数，不是没交接

`call_discoverable_user_tool` 是全场匹配率最低的动作：**310 次期望里 303 次不匹配（97.7%）**。但拆开看根因并不在交接本身：

```
270 / 303   agent 已经正确 give_discoverable_user_tool，但告诉用户的参数是错的
 33 / 303   工具本身没给对
```

也就是说，模型知道"该让用户自己去做"，也给对了工具，但**它口述给用户的参数（哪几笔交易、哪个账户）是错的**——这正是 F2（值错）和 F3（数量不全）经由自然语言传导到用户侧的结果。这条链路把 agent 的内部错误放大成了不可挽回的终态错误。

### F4 — 过度动作：动作全做对了，DB 仍然不匹配

有 11 条轨迹 `action_checks` 全部为 True，却拿 0 分。逐条查看多出来的调用：

```
task_005 ×3   多了一次 log_verification（验证了两遍，验证记录多写一条）
task_036      多调 file_credit_card_transaction_dispute_4829 3 次（重复立案）
task_063      多调 transfer_funds_between_bank_accounts_7291 4 次 + 多开一个户
task_102 ×1   多调 submit_referral 1 次
```

全量统计：485 条轨迹中 4 条调了两次 `log_verification`，90 条一次都没调。写类工具的重复调用（`close_debit_card_4721` 多调 15 次、`order_debit_card_5739` 多调 18 次、`close_credit_card_account_7834` 多调 10 次）是稳定存在的。

**含义**：在 DB 终态判分下，"多做"和"少做"等价致命。模型缺乏"这个写操作我是不是已经做过了"的幂等意识，尤其是在长对话里用户复述一次需求时会再执行一遍。

### F5 — 过早升级转人工

58 条失败轨迹以转人工收尾，其中 **27 条是期望动作里根本没有 transfer 的**（任务本可以自己完成）。典型：

> **task_025**：用户要为一笔 \$100,000 的企业采购申请商业信用卡。期望终态是用户自己调用 `apply_for_credit_card(card_type="Business Platinum Rewards Card", ...)`。模型的实际动作：`transfer_to_human_agents(reason="kb_search_unsuccessful_customer_requests_transfer")` —— **检索没找到就转人工**，而不是继续换角度检索。

policy 第 5 条写得很明确："Do this only if you absolutely have to, and you are sure that there are no potential actions you can take"。模型把**检索失败**误当成了**能力边界**。这与 §3 的检索打转是因果关系：检索反复无果 → 判定超出能力 → 转人工。

### F6 — 转人工的 `reason` 枚举选错

即使该转人工，65 次 transfer 里有 36 次参数不匹配，其中 20 次错在 `summary`、14 次错在 `reason`。对照期望与实际的取值分布：

| 期望的 reason | 模型实际填的 |
|---|---|
| `customer_demands_after_unavailable_offer_refusal` | `customer_frustrated_demands_human` (3) / `customer_requests_human_no_specific_reason` (1) |
| `unconfirmed_external_communication` | `kb_search_unsuccessful_customer_requests_transfer` (3) |
| `account_ownership_dispute` | `customer_requests_human_no_specific_reason` (3) |

模型自造并高频使用的 reason（`customer_frustrated_demands_human` 14 次、`customer_requests_human_no_specific_reason` 14 次、`technical_system_error` 9 次、`specialized_department_required` 8 次）**在期望集合里一个都不存在**。期望的只有 4 个场景码：`fraud_or_security_concern`、`kb_search_unsuccessful_customer_requests_transfer`、`customer_demands_after_unavailable_offer_refusal`、`account_ownership_dispute`、`unconfirmed_external_communication`。

**模型是按"用户的情绪状态"来选 reason 的（用户急了 → frustrated），而 KB 要求按"触发转接的业务场景"来选。** 这是典型的没去知识库查枚举取值、靠语义直觉填槽。task_008 和 task_014 两个任务就因为这一点 5 次全灭——模型转人工的时机完全正确，只有 reason 一个字段错。

### F7 — 凭空增加参数字段

47 处。模型给调用加上知识库没有的字段：

```
order_replacement_credit_card_7291  多加 expedited_shipping
order_debit_card_5739               多加 excess_replacement_fee / design_type / reason
```

这类错误还引发了全部 9 次工具硬报错中的 6 次（`order_debit_card_5739() got an unexpected keyword argument 'design_type'` 等）。模型对 discoverable 工具的签名是**半记忆半推测**的——policy 要求 unlock 之后读取真实签名，模型 unlock 了却没有以返回的签名为准。

## 3. 横切根因：检索行为失控

这是 F1/F5 的共同上游，值得单列。

### `grep` 的失败率是 47%

1091 次 `grep` 调用中 514 次返回 `No matches found`。对失败与成功的 pattern 做特征对比：

| 特征 | 失败 pattern (n=514) | 成功 pattern (n=577) |
|---|---|---|
| 含转义 `\.` | 4% | 1% |
| 纯单词 | 30% | 14% |

失败的 pattern 有清晰的形态：**跨行的过度约束**。

```
Gold Rewards Card.*2%.*dining|Gold Rewards Card.*2%.*travel|...
Silver Plus Account.*APY.*3\.0%.*4\.5%
order_debit_card_5739\(account_id
Platinum Account.*base APY|Diamond Elite.*base APY
```

模型把 `.*` 当成"文档内任意距离"来用，但 grep 是按行匹配的——`Gold Rewards Card` 和 `2%` 分处表格的不同行时永远匹配不上。它也会去 grep 函数调用的字面形式 `tool_name\(`，而 KB 里根本不是这样书写的。

另一端是相反的毛病：失败 pattern 里有 30% 是纯单词（成功组只有 14%），即模型猜一个 KB 里不存在的术语直接搜。两端合起来说明同一件事——**模型对 KB 的实际措辞和组织方式没有建立起表征**，只能在"过度具体"和"盲猜术语"之间来回试。

### BM25 检索的"同义反复"

真实轨迹（`simulations/10159306-...json`, task_096，求个人 Gold Plus 储蓄账户的 base APY）：模型连发 **15 次以上语义近乎相同的 `KB_search`**：

```
"Bronze savings account base APY rate"
"Bronze savings account base APY rate percentage"
"\"Gold Plus Account\" savings base APY rate"
"\"Gold Plus\" savings account base APY 3.5% 4.0% 5.0%"
"\"Gold Plus Account\" savings complete guide base APY"
"doc_savings_accounts_gold_plus_account_001 base APY"     ← 直接检索 doc id
"\"Gold Plus Account\" \"complete guide\" OR \"001\" OR \"base APY\" ..."  ← 塞入 OR 语法
```

每一次的 top-1 都是同一篇 `doc_bank_accounts_bank_accounts_(general)_012`。三个可观察的具体缺陷：

1. **对 BM25 无模型**。往查询里塞 `OR`、引号短语、具体百分比数字（`3.5% 4.0% 5.0%`）——这些对 BM25 只是稀释了有效 term 的额外噪声。
2. **不换检索轴**。15 次查询全部围绕 "Gold Plus + APY" 打转，从没尝试列举文档、从已获文档的交叉引用出发、或换成 grep 精确串。
3. **命名空间混淆**。个人 `Gold Plus Account` 与企业 `Gold Plus Saver Account` 反复串线，模型在 reasoning 里自己意识到了这个歧义却没有用它来构造更好的查询。

全局统计：485 条轨迹共 5638 次 `KB_search`，其中 134 次是**逐字重复**的查询（失败轨迹平均 0.33 次/条，通过轨迹 0.05 次/条，相差 6.5 倍）。逐字重复只是冰山一角——上面这种"换一两个词的同义重query"没法自动计数，但从样本看要普遍得多。

### 一个 reasoning 层面的异常

在多条轨迹里观察到 reasoning content 泄漏了未被解析的工具调用语法：

```
[think] <dots_function_call> <invoke name="KB_search"> <parameter name="query"> ... </parameter> </invoke> </dots_function_call>
```

模型在思维链里先写了一遍工具调用的 XML，然后才在正式的 tool_calls 字段里发出同样的调用。属于训练格式渗漏，本身不影响正确性（调用还是发出去了），但会浪费 token 且提示 reasoning/acting 的边界在该模型上不够干净。

## 4. 评测本身的问题（不应算到模型头上）

分析时发现若干任务的期望值存疑，建议在用这批数据做训练信号前先修：

1. **task_102 的 NL assertion 与对话事实矛盾**。断言要求"agent 应识别 Ember Analytics 超过 Sky Blue 的 4 年公司年限而不推荐它"，但对话中用户明确说 Ember Analytics 成立约 3 年。5 次 trial 全部判负，其中两次 judge 的 justification 自己指出了这个矛盾（"In fact, the conversation states Ember Analytics is about 3 years old, so the stated ineligibility reason is not supported by the dialogue"）。这是**任务数据的内部不一致**，5 条样本全是噪声。

2. **`transfer_to_human_agents.summary` 参与比对**。20 次不匹配错在 `summary`——这是一段自由文本，逐字比对没有意义。建议在 `compare_args` 里排除它（task_008 已经这样做了，只比 `reason`，但并非所有任务都是）。

3. **参数超集判负**。F7 里有相当一部分是模型填了额外的、值为 default 的字段（`expedited_shipping: false`、`excess_replacement_fee: 0`），业务语义上等价于不填。当前的严格 dict 比对把它们判负。是否要这么严可以再讨论，但至少在做失败归因时要把这类和真正的值错分开。

4. **0/1 判分掩盖了进步**。失败轨迹里期望动作匹配率的分布是相当平坦的（0.0: 46条, 0.2: 51条, 0.3: 50条, 0.5: 44条, 0.7: 44条, 0.8: 41条）——**有 53 条轨迹做对了 ≥80% 的动作却拿 0 分**。如果要把这批数据用于 RL 或者做能力追踪，建议同时记录 action-level 的部分分。

## 5. 优先级建议

按"影响面 × 可修复性"排序：

| 优先级 | 方向 | 针对 | 预期收益 |
|---|---|---|---|
| P0 | **先取全集再动手**：把"列出该用户全部账户/卡/相关交易"作为写操作前的强制前置步骤 | F1, F3, F3b | 覆盖最大的两类失败；影响 173 条失败轨迹 |
| P0 | **枚举取值必须来自 KB**：`reason`、`dispute_category`、`card_action`、`account_class`、`card_design`、`delivery_option` 这些槽位禁止凭语义直觉填，必须能指到 KB 出处 | F2, F6 | task_008/014 这类"只错一个字段"的全灭任务可直接翻盘 |
| P1 | **检索策略**：单个事实连续 3 次检索无果时强制换轴（列目录 / 顺着已获文档的引用走 / 改用单 term grep），禁止同义重 query；grep pattern 限制为单行可匹配的短串 | §3, F5 | 直接削减 F5 的 27 条无端转人工，并缓解失败轨迹 1.5 倍的 token 开销 |
| P1 | **写操作幂等检查**：每次写之前回看本轮已执行的写操作清单 | F4 | 11 条"动作全对仍 0 分"可直接转正 |
| P2 | **政策算术的显式化**：责任上限分档、临时信用资格、APY 叠加、补款金额这几类，要求先写出适用的规则条款再代入数值 | F2 子模式 1/2 | dispute 族（189 条，pass 0.12）是最大的失败池 |
| P2 | **纠正保守偏置**：默认档不是安全选项。`PREMIUM`/`EXPEDITED`/`cancel_and_reissue`/`expedited=true` 在政策命中时是唯一正确答案 | F2 子模式 3 | — |
| P2 | **交接给用户前复核参数**：口述给用户的 ID 和数量是最终答案，且不可挽回 | F3b | 303 次失败的用户工具调用中 270 次源于此 |

---

*分析脚本与中间产物：`/tmp/claude-0/.../scratchpad/`（会话级临时目录，未入库）。所有数字均由 `results.json` + `simulations/*.json` 485 条轨迹全量统计得出，非抽样。*
