# tau3-banking 常见失败模式总结（run: `tau3_banking_dots3_omni_preview_gw134_1trial`）

对 `data/simulations/tau3_banking_dots3_omni_preview_gw134_1trial/` 下 97 条完整轨迹（97 个任务，各 1 trial）做的全量统计分析，不与其他评测结果比较，只总结这次运行本身暴露出的失败模式。

## 0. 数据口径

| 项 | 值 |
|---|---|
| 被测 agent | `openai/dots3_omni_preview`（`reasoning_effort: medium`，`temperature 1.0`） |
| user simulator | `openai/gpt-5.4-mini`（`temperature 0`，`reasoning_effort: medium`） |
| domain | `banking_knowledge` |
| 规模 | 97 个任务 × 1 trial = 97 条 simulation |
| 预算 | `max_steps 200`，`max_errors 10` |
| 奖励口径 | 87 条 `DB`（终态数据库比对）、9 条 `ACTION`、1 条 `DB + NL_ASSERTION`；0/1 判分，无部分分 |

## 1. 总体表现

```
pass@1              18.6%   (18/97)
termination_reason   全部 user_stop（97/97，无一次撞 max_steps）
工具调用总数          2564
硬错误（Python 层报错）117 次，分布在 48/97（49.5%）条轨迹中
```

**模型几乎不会"卡死"——它总是自信地走完流程，给出一个终态，无论对错。** 97 条轨迹没有一条以 `max_steps` 收场，全部是模型自己或用户判断"事情办完了"而正常结束对话。失败因此几乎都不是"没做完"，而是"做完了但做错了"——这是理解下面所有失败模式的前提：**多数失败在对话表面完全看不出来**，用户会正常道谢挂机，只有拿终态数据库去比对才会发现出了问题。

### 失败轨迹更长、更贵

| | 通过 (n=18) | 失败 (n=79) |
|---|---|---|
| 平均消息数 | 37.2 | 60.7 |
| 平均工具调用数 | 16.2 | 27.9 |
| 平均 `KB_search` 次数 | 8.8 | 12.1 |
| 平均 `grep` 次数 | 1.8 | 2.4 |

失败轨迹的工具调用量接近通过轨迹的 1.7 倍，检索量接近 1.4 倍。更多的检索和动作不代表模型在有效地补足信息——很大一部分只是在打转、试错、或者被硬错误打断后重试（见 §2.1）。

### 通过率随任务动作数急剧衰减

| 期望动作数 | 任务数 | pass@1 |
|---|---|---|
| 1–4 | 30 | 0.43 |
| 5–9 | 29 | 0.14 |
| 10–14 | 12 | 0.08 |
| 15–19 | 13 | 0.00 |
| 20–24 | 7 | 0.00 |
| 25–29 | 4 | 0.00 |
| 30+ | 2 | 0.00 |

**15 个动作以上的任务（26 个）本次全灭，一次都没通过。** 简单的 1–4 步咨询/推荐类任务通过率还能到 43%，一旦任务需要多步骤、多对象的写操作链条，通过率迅速跌到 0。0/1 判分下，任意一步出错整条轨迹归零，这个曲线本质上反映的是**单步正确率的复合效应**：假设每一步独立正确率在 85%–90% 左右，20 步左右的任务综合通过率自然趋近于 0。

### 按业务族拆分

| 业务族 | n | pass@1 |
|---|---|---|
| 争议/退单 dispute | 38 | 0.05 |
| 账户开立/关闭 | 19 | 0.11 |
| 借记卡操作 | 6 | 0.00 |
| 推荐 referral | 7 | 0.14 |
| 账户补款 statement/account credit | 8 | 0.00 |
| 其他（多为 1–2 步的咨询/推荐类） | 19 | 0.68 |

**争议/退单是最大也是最差的失败池**（38 个任务，仅 5% 通过），且平均期望动作数最高（14.5 步）——这类任务典型地需要"拉全量交易记录 → 逐笔判断是否符合纠纷条件 → 逐笔提交纠纷 → 套用正确的责任分档规则"，链条长、每一步都有出错空间，是下面几乎所有失败模式的重灾区。

## 2. 失败模式总览

对 97 条轨迹的 `action_checks`（合计 955 个期望动作）和全部工具调用做了展开统计，识别出以下几类失败模式：

| # | 失败模式 | 规模 |
|---|---|---|
| M1 | 工具调用触发机制性硬错误（协议违反 / 签名不匹配 / 幻觉工具名 / 类型错误） | 117 次，48/97 条轨迹 |
| M2 | 期望的业务工具从未被调用 | 38/97 条轨迹 |
| M3 | 混淆相似工具，执行了"另一个"看似合理的写操作 | 至少 1 条清晰案例 |
| M4 | 参数值错误，集中在需要按政策推算的字段 | 144 处字段级不匹配 |
| M5 | 枚举不全：该做 N 次的动作只做了 M 次 | 72 处 |
| M6 | 过度动作：动作全对但终态仍不匹配 | 2 条 |
| M7 | 过早升级转人工 | 6 条轨迹（5 条致命） |
| M8 | 转人工 `reason` 枚举选错 / 自造枚举值 | 8/14 次 transfer 用了不存在的枚举值 |
| M9 | 完全不作为：任务需要写操作，一次都没做 | 10 条 |

以下逐条展开，按对整体失败面的贡献从大到小排列。

### M1 — 工具调用机制性硬错误（本次最大的单一问题）

117 次工具调用直接返回 Python 层报错，分布在近一半（48/97）的轨迹里。按内容分类：

| 错误类型 | 次数 | 影响轨迹数 | 示例 |
|---|---|---|---|
| **未 unlock 直接 call**（违反"先 unlock 后 call"协议） | **53** | **34/97（35%）** | `Tool 'get_user_dispute_history_7291' has not been unlocked...` |
| 业务规则正常拒绝（非模型 bug，是合理的业务反馈） | 24 | — | `Account eligibility requirements not met.`、`Insufficient funds...` |
| 参数签名不匹配（多传/少传关键字参数） | 17 | 13 | `order_debit_card_5739() missing 4 required positional arguments...` |
| 幻觉工具名（KB 里根本不存在的工具） | 17 | 11 | `Unknown agent tool 'downgrade_credit_card_3847'.` |
| 未处理的类型错误（模型传字符串，工具内部按数字比较） | 6 | 1 | `'<=' not supported between instances of 'str' and 'int'` |

**光是"没 unlock 就直接 call"这一种错误就占了硬错误总量的 45%，出现在 35% 的全部轨迹里。** domain policy 明确要求 discoverable 工具必须先 `unlock_discoverable_agent_tool` 才能 `call_discoverable_agent_tool`，模型系统性地跳过这一步，直接调用真实工具名，被环境拒绝后再回头补 unlock——这本身能恢复，但会白白浪费几轮对话和 token，并让模型看起来像是在"反复试错"而不是按协议行事。统计上，触发过"未 unlock"错误的 34 条轨迹 pass@1 只有 8.8%（3/34），明显低于全场 18.6% 的基线。

**还有一层更底层的基础设施问题**：这个模型用自己的原生格式 `<dots_function_call>...</dots_function_call>` 输出工具调用，日志里出现了 79 次"网关未能把原生格式转译成标准 tool_calls，靠正则兜底捞取"的补救记录；其中 6 次补救彻底失败，模型的输出既没有 `content` 也没有 `tool_calls`，直接抛出异常，迫使涉及的任务（`task_008`、`task_048`、`task_063`、`task_079`、`task_088`、`task_096`）**整体重跑**——有的任务在重跑前已经跑了 1000 多秒，进度全部作废。这 6 个任务最终全部失败。

一个典型的复合案例，`task_086`（借记卡交易纠纷）：

```
call file_debit_card_transaction_dispute_6281(..., disputed_amount: '300.0',
     card_in_possession: 'true', provisional_credit_eligible: 'true', ...)
→ Error: missing 1 required positional argument: 'customer_max_liability_amount'

补上该字段，仍然全部按字符串传值:
call file_debit_card_transaction_dispute_6281(..., customer_max_liability_amount: '500')
→ Error: '<=' not supported between instances of 'str' and 'int'

第三次尝试，模型把值又包了一层引号:
call file_debit_card_transaction_dispute_6281(..., customer_max_liability_amount: '"500"')
→ Error: '<=' not supported between instances of 'str' and 'int'   (依旧失败)
```

模型**几乎所有工具参数都以字符串形式传递**（`'true'`/`'false'`/`'300.0'` 而非原生布尔/数字），而工具实现在拿到错误类型时抛出的是未处理的 Python `TypeError`，不是一条对模型可读的校验错误信息——模型看到裸异常后不知道怎么修，只能盲目换一种转义方式再试，完全没有触及问题本质。

### M2 — 期望的业务工具从未被调用

38/97 条轨迹里至少有一个期望的业务工具（已按 `call_discoverable_agent_tool` 的 `agent_tool_name` 参数展开为真实工具名）从未出现过：

```
21  submit_cash_back_dispute_0589
15  close_debit_card_4721
11  order_debit_card_5739
 9  open_bank_account_4821
 9  get_bank_account_transactions_9173
 7  update_transaction_rewards_3847
 7  unfreeze_debit_card_3893
 6  file_credit_card_transaction_dispute_4829
 6  get_all_user_accounts_by_user_id_3847
 6  log_verification
 6  get_pending_replacement_orders_5765
 6  approve_credit_limit_increase_5847
```

榜首 `submit_cash_back_dispute_0589`（返现纠纷提交）领跑，说明返现类争议任务里模型经常连第一步的正式提交流程都没走到。

**读类工具存在明显的"发现惰性"**：标准工具 `get_credit_card_accounts_by_user` 被多调 32 次（期望只有 3 次），`get_credit_card_transactions_by_user` 完全没有被期望过却被调用了 34 次。模型偏好它"熟悉"的通用工具，而不是去主动发现场景专属的 discoverable 工具——即使 KB 里已经有更贴合任务的专用工具。

### M3 — 混淆相似工具，执行了"看起来对"的错误写操作

`task_020`：用户怀疑自己的信用卡返现算错了，期望流程是 agent 把 `submit_cash_back_dispute_0589` 这个**用户侧**工具交给用户，让用户对每一笔错误交易各提交一次正式纠纷。实际发生的是：

```
assistant 调用 update_transaction_rewards_3847(transaction_id=..., new_rewards_earned="1600 points")
  （连续 4 次，分别针对 4 笔交易，全部执行成功，无报错）
assistant: "All 5 updates are confirmed in your transaction history. Here's the final summary: ..."
user: "Thanks, that's all I needed. ###STOP###"
```

`update_transaction_rewards_3847` 是另一个真实存在、agent 侧可直接调用的工具（直接改写奖励记录），跟策略要求的"走正式纠纷流程、由用户自己提交"是两条完全不同的业务路径。模型选错了工具，但**因为选的工具本身有效、调用会成功、还生成了一张像模像样的"已确认更新"对比表**，从对话表面看不出任何异常——用户满意地道谢挂机，实际终态完全不对，reward=0。这是本次运行里"静默错误完成"最干净的一个样本：**不是没做事，是做了一件看起来对、实际上文不对题的事**，比完全不作为更危险，因为连事后复盘都不容易一眼看出问题。

### M4 — 参数值错误：集中在需要按政策推算的字段

对每一个"工具被调用但没匹配上"的期望动作做字段级 diff（严格按各任务的 `compare_args` 限定字段），出错最多的字段：

```
12×  file_debit_card_transaction_dispute_6281.customer_max_liability_amount   期望 50    → 填 500
11×  file_credit_card_transaction_dispute_4829.eligible_for_provisional_credit 期望 false → 填 true
10×  submit_cash_back_dispute_0589.transaction_id   期望具体交易号 → 完全没填/漏填
 9×  log_verification.time_verified                期望精确到分钟的模拟时钟值 → 瞎填/截断
 6×  open_bank_account_4821.account_class           档位选错
 4×  apply_for_credit_card.card_type                推荐的信用卡种类不对
 4×  file_debit_card_transaction_dispute_6281.transaction_type    pin_purchase 被判成 signature_purchase/online_purchase
 4×  submit_referral.account_type                   推荐的账户档位不对
 3×  file_credit_card_transaction_dispute_4829.phone  '720-555-0348' → 被加上国家码 '+1-720-555-0348'
 3×  file_credit_card_transaction_dispute_4829.transaction_id  'txn_754027c485bf' → 丢掉前缀变成 '754027c485bf'
```

三个可辨认的子模式：

1. **分级规则套错档**。`customer_max_liability_amount` 50→500 是最典型的：卡还在手上、及时上报的责任上限本应是 \$50 档，模型套成了 \$500 档。这类错误往往意味着模型没有把"条件"和"档位"对齐——判断走了一半，最后套值时用了错误的分支。
2. **保守/默认偏置**。`eligible_for_provisional_credit` 11 次错误全部是 false→true（过度给付临时信用），这是一种系统性的"往宽松方向猜"的倾向：模型宁愿多给用户一点好处，也不愿意在没有十足把握时判定不给。
3. **精确匹配失败暴露的事实性幻觉**。`log_verification.time_verified` 的期望值是同一个模拟时钟常量，模型本应调用 `get_current_time`（本次被调用 84 次）取到这个值原样填入。抽查 `task_021` 发现：轨迹里 `get_current_time` 正确返回了当前时间，而同一条轨迹里 `log_verification` 却被填成了客户**出生日期的月日拼上一个瞎编的年份**——模型拿到了正确的工具返回值，下一步却写入了一个跟当前时间毫无关系的捏造值。这不是格式误差，是一次清楚的事实替换错误。

另外还有两处格式漂移值得注意：电话号被自作主张加上 `+1` 国家码前缀，交易号被自作主张去掉 `txn_` 前缀——模型在"整理格式"，而不是编不出正确值,是一种可教的、模式清晰的错误。

### M5 — 枚举不全：找不齐所有符合条件的对象

72 处"参数格式对但调用次数不够"。领跑的是 `get_all_user_accounts_by_user_id_3847`（13 处少调）、`submit_cash_back_dispute_0589`（7 处）、`file_credit_card_transaction_dispute_4829`（7 处）——同一个病根：**模型没有先把对象全集拉出来，就开始逐个处理**，于是只处理了对话中被提到的那一两个，漏掉了数据库里其余符合条件的对象。这和 M2 是同一类问题的两种表现：M2 是完全没找到工具入口，M5 是找到了但没用够。

### M6 — 过度动作：动作全对仍不通过

样本下只有 2 条，但很能说明问题：`task_063` 的 `action_checks` 全部为 True，但终态数据库比对不通过——说明模型在完成期望动作之外还额外做了写操作，污染了终态。在 0/1 判分下，"多做"和"少做"一样致命；`task_063` 同时也是因为网关解析异常被整体重跑过的任务之一，是本次样本里最"脏"的一条，叠加了基础设施问题和过度写操作两种失败模式。

### M7 — 过早升级转人工

6 条轨迹在没有被期望 transfer 的情况下调用了 `transfer_to_human_agents`，其中 5 条最终失败（占 79 条失败轨迹的 6.3%）。典型触发点是身份验证失败或检索没能立刻找到答案——模型把"暂时受阻"当成了"能力边界"，选择转人工而不是换个方式继续尝试。

一个反例值得记录：`task_025`（企业客户申请商业信用卡）里，agent 因身份验证失败转人工，本应算作策略违规，但因为该任务的评分口径完全依赖用户自己独立调用 `apply_for_credit_card`，与 agent 是否转人工无关，最终 reward 仍为 1.0。这提醒我们：**"过早转人工"是确定的行为缺陷，但不一定总是拖累最终得分**，量化其危害时应以最终 reward 是否受影响为准，而不是简单统计转人工次数。

### M8 — 转人工 `reason` 枚举选错

本次运行 agent 共发起 14 次 transfer，实际使用的 `reason` 值：

```
3×  customer_frustrated_demands_human          ← 不存在的枚举值
3×  specialized_department_required            ← 不存在的枚举值
2×  customer_requests_human_no_specific_reason  ← 不存在的枚举值
2×  kb_search_unsuccessful_customer_requests_transfer   （有效枚举）
1×  unconfirmed_external_communication          （有效枚举，用对了）
1×  account_closure_request                     ← 不存在的枚举值
1×  request_completed_customer_wants_human_followup  ← 不存在的枚举值
1×  technical_system_error                      ← 不存在的枚举值
```

**14 次里有 8 次用的是 KB 枚举表里根本不存在的值**。这些自造词全部命中同一个规律：模型是**按用户的情绪状态或问题的"感觉严重程度"**去选 reason 的（"客户不耐烦了" → `customer_frustrated_demands_human`，"这事听起来需要专门部门" → `specialized_department_required`），而不是照着 KB 规定的、基于触发场景的固定枚举去选。这是典型的"靠语义直觉填槽，而不是去查真实取值表"。

### M9 — 完全不作为

10 条轨迹里，任务明确需要至少一次写操作，模型一次都没做。除了信用卡申请类任务（这类由用户自己触发，不完全是 agent 的责任）之外，`submit_cash_back_dispute_0589`（返现纠纷）和 `file_credit_card_transaction_dispute_4829`（信用卡交易纠纷）是最常见的"完全没提交"的场景，说明返现/信用卡纠纷这条业务线，模型除了"提交不全"（M5）之外，还有相当比例是"压根没走到提交这一步"。

## 3. 检索行为

```
grep 总调用 223 次，"No matches found" 47 次，失败率 21.1%
KB_search 总调用 1115 次，逐字重复查询仅 2 次（0.2%）
```

`grep` 失败样本呈现出清晰的两类误用：

```
(?i)card (blocked|declined) (when|abroad|travel)      -- 假设短语会连续出现在同一行，但 grep 按行匹配
## Platinum Rewards Card                                -- 直接搜 Markdown 标题字面值，命中率随文档排版而定
tool: get_cbd|get_.*cash_back_dispute|cash_back_disputes_.*tool   -- 试图搜索"工具名的字面写法"
unlock_discoverable_agent_tool\(\".*apply.*\"\)         -- 把函数调用语法当成 KB 文本去搜
get_(cbd|transaction|account|dispute|card)_by_[a-z_]+   -- 用代码变量命名习惯去猜 KB 文档措辞
```

一类是**把跨行内容当成同一行**（`.*` 被误用为"文档内任意距离"，但 grep 是逐行匹配的），另一类是**把函数调用语法本身当成 KB 里会出现的文本**去搜——KB 文档是自然语言写的政策说明，不会包含 `tool_name(...)` 这种字面调用语法。

KB_search 本身的逐字重复率很低（0.2%），没有观察到"同一个概念反复变换措辞死磕"的病态重复模式，检索预算总量（通过轨迹 8.8 次/条，失败轨迹 12.1 次/条）合理，说明检索不是本次最大的瓶颈——真正的瓶颈在于拿到信息之后，如何把信息正确地转成工具参数（见 M4）以及是否守住调用协议（见 M1）。

## 4. 评测数据本身的问题

分析时发现两处不应算到模型头上的问题：

1. **`task_102` 的 NL assertion 与对话事实矛盾**。断言要求"agent 应识别 Ember Analytics 超过 Sky Blue 的 4 年公司年限而不推荐它"，但对话中用户明确说 Ember Analytics 成立约 3 年。judge 的 justification 也自己指出了这个矛盾："the conversation states Ember Analytics is about 3 years old, so it does not exceed Sky Blue's 4-year age limit anyway." 这是任务数据本身的内部不一致，建议在用这条数据做训练信号前先修正或剔除。
2. **`log_verification.time_verified` 的精确匹配设计**可能引入不必要的噪声。这个字段要求逐字匹配一个精确到秒的模拟时钟字符串。虽然确实抓到了一例模型的事实性幻觉（见 M4），但也有几例更像是格式/精度问题（只填了日期没填时间、时分对不上）而非理解问题，值得区分对待。
3. **0/1 判分掩盖了部分正确的努力**。失败轨迹里动作匹配率的分布相当平坦（0.0: 11条，0.2: 12条，0.5: 7条，0.8: 12条……），**11/79 条轨迹（13.9%）做对了 ≥80% 的动作却仍然拿 0 分**。如果要把这批数据用于 RL 或能力追踪，建议同时记录 action-level 的部分分，而不是只看最终 0/1。

## 5. 优先级建议

按"影响面 × 可修复性"排序：

| 优先级 | 方向 | 针对 | 预期收益 |
|---|---|---|---|
| P0 | 修复网关对该模型原生工具调用格式（`<dots_function_call>`）的转译，避免整任务重跑、避免依赖正则兜底 | M1 | 消除 6 个任务的整体重跑成本 |
| P0 | 重申"先 unlock 再 call"的强约束：考虑让编排器在收到未 unlock 的 call 时直接自动纠正，而不是仅报错等模型自己反应过来 | M1 | 硬错误里 45% 的量，影响 35% 的轨迹 |
| P0 | 先取全集再动手：把"列出该用户全部账户/卡/相关交易"作为写操作前的强制前置步骤 | M2, M5 | 覆盖最大的一类失败 |
| P0 | 枚举取值必须来自 KB：`reason`、`dispute_category`、`card_action`、`account_class` 等槽位禁止凭语义直觉填，必须能指到 KB 出处 | M4, M8 | 8/14 次转人工的错误可直接消除 |
| P1 | discoverable 工具的参数做类型强校验/自动转换，而不是让内部比较运算符直接抛出未处理异常 | M1（`task_086` 案例） | 减少"模型看不懂报错、盲目重试"的死循环 |
| P1 | 检索策略：`grep` pattern 限制为单行可匹配的短串，避免把函数调用字面语法或跨行假设当成有效 pattern | §3 | 降低 21% 的 grep 失败率 |
| P1 | 写操作幂等检查 / 过度动作检测 | M6 | 样本小但代价明确 |
| P2 | 政策算术的显式化：责任上限分档、临时信用资格等，要求先写出适用规则条款再代入数值 | M4 | 争议族（38 个任务，pass 0.05）是最大失败池 |
| P2 | 相似工具去歧义：`submit_cash_back_dispute_0589` vs `update_transaction_rewards_3847` 这类语义相近但流程不同的工具，在工具描述/发现阶段加更强的区分提示 | M3 | 样本小但危害大，是"静默错误完成"里最难被发现的一类 |
| P2 | 修正 `task_102` 的 NL 断言 | §4 | 确定性 bug |

---

*所有数字由 `results.json` + `simulations/*.json` 97 条轨迹全量统计得出，非抽样。*
