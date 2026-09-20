# 世界合成 pipeline（worldgen）——通俗说明

> 本文保留旧版流水线说明。旧格式的发布标记已不能作为验证证书，生产入口已切换到 [Worldgen V2](worldgen-v2.md)：声明式品类、独立任务与文档审核、反例验证及证书发布。旧命令仅用于显式开发检查。

这份文档讲的是 `src/tau3/worldgen/`：一条**造一个全新银行世界**的流水线。

先说清楚它和仓库里另一条流水线的区别，这是最容易混淆的地方：

| | `src/tau3/synthesis/`（tau3-AA） | `src/tau3/worldgen/`（本文） |
|---|---|---|
| 世界 | **钉死**官方 `banking_knowledge` | **新造**一个 |
| 合成什么 | 客户、账户状态、目标、参考动作 | 知识库文档、数据库、任务 |
| 知识库 | 官方那 698 篇，一字不改 | 全新生成 |
| 用途 | 产 SFT / RL 训练任务 | 防基准被参数化记忆污染、可控扩量 |

两者目标正交，不能互相替代。

---

## 一、为什么要造新世界

官方 τ-Banking 基准的工具名长这样：`open_bank_account_4821`、`get_closure_reason_history_8293`。那四位数字后缀是**故意的**——模型猜不出 `8293`，只能真的去知识库里检索到写着它的那篇文档。这是基准衡量"检索能力"而非"背诵能力"的关键机制。

但只要这套语料公开得够久，模型就可能把 `8293` 背下来。届时基准测的就不再是检索了。

造一个新世界，就是把这层防线重新立起来：新的产品、新的文档、新的后缀。

---

## 二、这条流水线做什么、不做什么

**做**：知识库文档、`db.json`、任务集。

**不做**：不碰工具实现。官方那 48 个可发现工具 + 20 个常驻工具的**代码原样复用**，只在运行时给它们换个名字（后缀重掷）。

为什么这么定？因为工具语义是环境的一部分，重写它们等于重写半个基准。复用官方实现意味着：

- 合成世界与官方世界的 agent 行为**可比**（同一套工具、同一套检索、同一份 policy 模板）；
- 工具里那些真实的前置条件（开储蓄户要求支票户满 14 天等）**自动继承**，不用重新实现；
- 唯一新增的是一层"别名"：`close_bank_account_7392` 在新世界里叫 `close_bank_account_9480`，agent 只能通过检索文档才知道。

---

## 三、一个世界是怎么造出来的

十个阶段，每个都是独立命令、可断点续跑。

```
calibrate → init-world → build-schema → assign → map-tools
          → plan-docs → render → build-tasks → lint → verify → accept → publish
```

### 阶段 0：calibrate —— 先量靶子

从**本仓库实际安装的**官方语料现场测出基线：698 篇文档、71 个主题、97 个任务、每篇平均 257.8 token、每个任务平均 9.89 篇 gold 文档。

**为什么不直接抄论文表？** 因为对不上。论文说 gold 文档均值 18.6，实测是 9.89；论文说每篇 278.7 token，实测 257.8。后续所有门禁读的是实测值，论文数字只作为注释保留。

### 阶段 A1：build-schema —— 让模型出"骨架"，不出"值"

模型只被允许输出属性的**形状**：名字、类型、单位、合理区间。**绝不输出任何具体数值**。

这是整条流水线最重要的一条纪律。模型一旦自己编数字，文档里的费率就和数据库对不上，任务直接变成不可解，而且很难发现。让它从头到尾看不见数值，它就无从编起。

### 阶段 A2：assign —— 用求解器填值，并埋三类陷阱

值由一个确定性求解器赋，不是采样试错。它保证：

1. **答案全局唯一**——客户说完需求后，整个目录里**只有一个**产品满足全部条件；
2. **两个"差一点"的干扰项**——各自恰好违反一条约束。为什么恰好一条？如果一个产品同时违反两条，agent 看一篇文档就能排除它，那道比较题就白出了；
3. **三类陷阱**：
   - *促销陷阱*：广告写着"+0.5% 加息"，但它的基准利率低了 0.75%。跟着促销走的 agent 必错；
   - *时效陷阱*：宣传力度最大的那张卡，促销窗口**已经过期**。不调 `get_current_time()` 就会选错；
   - *近似干扰项*：上面第 2 条。

### 阶段 B：map-tools —— 重掷后缀

给每个带四位后缀的工具算一个新名字，写进世界的 `alias_map`。同一个 base 下的三个 `activate_debit_card_*` 会得到三个不同的新后缀；不同 base 之间**允许**撞后缀（官方语料本来就有这种巧合，这正是让后缀无法用来推断工具族的原因）。

### 阶段 C：plan-docs —— 决定"哪篇文档讲什么"

这一步决定检索有多难，是整条流水线的难度旋钮所在。

- **覆盖**：每个变量至少出现在一篇文档里；
- **碎片化**：平均每个变量出现在 1.35 篇文档里——高于 1 是为了单次漏检不致命，远低于 2 是为了漏检不免费；
- **反聚合**：任何一篇文档透露的某任务知识不得超过 `⌈|Req|/leak_divisor⌉`。这条直接决定一个任务需要读多少篇文档。

### 阶段 D：render —— 两段式写作

**第一遍（模型）**：给它属性的名字，让它写散文，所有数字必须写成 `[[var_copper_monthly_fee]]` 这样的占位符。

**硬门禁**：输出里出现任何占位符之外的裸数字，直接判为幻觉、重写（白名单只有列表序号、四位年份、工具名里的数字）。

**第二遍（确定性）**：把占位符换成求解器赋的值，把 `[[tool:...]]` 换成本世界的别名。

风格上会在**两个模型 × 四种人设 × 四种语域**之间轮转，避免整个语料被 embedding 按作者聚类——那会让检索变得不真实地容易。

### 阶段 F：build-tasks —— 出题和铺数据

生成任务与 `db.json`。几条关键纪律：

- **参考动作必须走 agent 真实的工具接口**（`unlock` → `call`），这样官方评估器记录的审计日志才和真实轨迹一致；
- **客户只描述需求，绝不点名产品**。这一条是实测出来的：点名产品时检索太容易（recall 0.304，官方 0.069），不点名则偏难（0.032）；
- **背景噪声用户**：除任务客户外再铺几百个不参与任何任务的客户，否则 agent 不用核对身份也能"猜中"唯一匹配的记录。

### 阶段 E / G：lint 与 verify —— 门禁

`lint` 跑结构与文本检查（唯一性、干扰项、单文档泄漏、悬空引用、近重复、工具孤儿、命名冲突、可检索性）。

`verify` 是两道更重的关：

- **G1 可解性**：拿参考动作到**真环境**里重放。官方工具会自己拒绝不合规的操作（返回 `Error:` 文本），所以顺序排错的任务在这里就会失败；
- **G2 gold 极小覆盖**：任务需要的知识必须被 gold 文档集**恰好**覆盖——少一篇则任务不可解，多一篇则检索分数失去意义。

### 阶段 H：accept —— 与官方语料比

三层，越往后越有权威：

1. **文本层**（参考性）：长度、句长、词汇多样性、表格/项目符号占比；
2. **检索层**（阻断）：同一个 BM25 管道下，任务的 gold 文档有多少能被一轮检索捞到；
3. **行为层**：同一个 agent，**同时**跑合成世界和官方任务的等量对照样本，比**差值**。

第 3 点必须强调：论文里那些 pass^1 数字来自 GPT-5.2 / Opus，和本仓库能用的模型不可比。唯一有意义的问题是"同一个 agent 在两边表现差多少"。

### 阶段 I：refresh —— 改一个值不用重写

模板哈希**刻意不含值**。所以：

- 改一个费率 → 只重跑第二遍填充，**零次模型调用**，毫秒级；
- 改了文档的"依据"（变量进出、换标题、换作者）→ 哈希变了，报 `needs_model`，必须重渲。

刷新过的世界会**自动退回 draft**，`load_world` 随即拒绝加载，直到重新 verify + publish。

---

## 四、怎么用

```bash
W=data/synthetic/tau3-world/demo

uv run tau3 worldgen calibrate    --output data/synthetic/tau3-world/targets.json
uv run tau3 worldgen init-world   --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen build-schema --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen assign       --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen map-tools    --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen plan-docs    --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen build-tasks  --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen render       --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen lint         --world $W --config configs/worldgen/tau3-mid.yaml
uv run tau3 worldgen verify       --world $W
uv run tau3 worldgen accept       --world $W --config configs/worldgen/tau3-mid.yaml --gate all
uv run tau3 worldgen publish      --world $W
```

跑完之后，这个世界就是一个可用的 domain：

```bash
TAU3_SYNTH_WORLD=$W OPENAI_API_BASE=http://10.39.62.231:9091/v1 OPENAI_API_KEY=not-required \
  uv run tau3 run --domain banking_synth \
  --agent-llm openai/gemini-3.5-flash --user-llm openai/gemini-3.5-flash
```

三份配置：`tau3-mini.yaml`（10 主题 / 80 文档，打通用）、`tau3-mid.yaml`（40 / 400）、`tau3-full.yaml`（71 / 698，对齐官方规模）。**规模是配置项，不是改代码**。

---

## 五、成本（实测）

一个 400 文档 / 38 产品的世界：

| 阶段 | 总 token | 单位成本 |
|---|---|---|
| A1 生成骨架 | 266,856 | 7,023 / 产品 |
| D 渲染文档 | 979,465 | 2,449 / 文档 |

**每篇文档 ≈ 3,116 token**，128 并发下约 **3.4 秒/篇**。400 篇约 23 分钟，698 篇约 40 分钟。

金额是零——端点自托管、不按 token 计费。真实成本是 token 和时间。

吞吐**约 6–7 万 TPM，且与并发设置无关**：瓶颈在端点，不在客户端。把 concurrency 从 16 提到 64 不会让它更快。

---

## 六、现在做到了什么、还差什么

**已达成**（40 主题 / 400 文档的世界）：渲染 400/400 零失败、Lint 零错误、**Verify 50/50 全部可解且 gold 集极小**、横向扩展验证到 10,000 篇文档。

**还没达成**，三项，都已定位到原因：

1. **检索层未通过**：合成 recall@10 = 0.222，官方 0.069，仍易约 3.2 倍；
2. **gold 均值 6.4，官方 9.89**；
3. **gold 复用 8.42，官方 4.40** —— distinct gold 文档只有 38 篇。

第 3 点是前两点的共同根因：目前**每个产品品类只有一个 scenario family**，所以同品类的所有任务答案相同、证据相同。为什么不能简单加家族？因为两个家族都覆盖同一目录时会争抢同一批变量、互相覆盖答案（实测发生过）。正确解法是**按产品分段划分家族**（官方语料的 Entry/Mid/Premium 分层就是这个思路），这是同时改善多样性、gold 数和检索难度的同一个杠杆。

**另有一处已知浪费**：A1 阶段占总成本 21% 且响应会截断——它在一次 strict-JSON 响应里要 17 个变量对象，撞上端点 8,192 token 的输出上限，靠翻倍预算重试才成功。拆成两批可削掉大部分。

---

## 七、贯穿全线的几条纪律

1. **门禁必须会拒绝。** 每条检查都配了负向测试：把排序任务的关户排到开户之前，G1 必须报错；gold 集少一篇必须报出缺失的知识元素；多一篇必须报出冗余项。只会通过的门禁没有价值。
2. **失败要响亮。** 变量没赋值 → 报错而非留 `None`；规则环境不再执行 → 拒绝生成；产品名撞车 → 抛异常而非静默碰撞。
3. **数字要现场量。** 论文表、我的直觉、看起来合理的估计，都不作数。
4. **官方语料只读。** 整条流水线对 `data/tau3/domains/banking_knowledge/` 只读，有测试盯死。

相关文档：[tau-banking-synth.md](tau-banking-synth.md)（设计依据）、[task-synthesis.md](task-synthesis.md)（另一条流水线）。
