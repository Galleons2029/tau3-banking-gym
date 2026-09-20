# Worldgen V2：可验证、可扩展的种子世界

V2 将产品事实、动态表、操作、任务目标和证据依赖放在同一个可执行契约中。生成器可以提出候选内容，但只有经过验证的产物才能发布；不确定、预算耗尽、模型复核失败均不会自动视作通过。

这里的“正确”首先指虚构银行世界内部的事实、文档、操作和任务一致。它不等于真实银行产品资料，也不构成与论文任务分布、自然度、难度等价的证明。

当前定位为可靠种子生成的基础版本。继续扩展前需通过单独的盲解、真实评分反例和完整在线覆盖准入；`expand-v2`／`propose-category` 默认需要 `--readiness`。已有基础开发命令可以显式使用 `--foundation-only`，其产物不具有扩产资格。详见 [独立验证与受控扩展准入](worldgen-v2-independent-validation.md)。

## 快速使用

```bash
# 生成普通 JSON 品类包，可保存在版本控制中。
uv run tau3 worldgen scaffold-category \
  --id term_deposits --profile term_deposit \
  --output /tmp/term_deposits.json

# 确定性文本无需模型，可离线生成和完整验证。
uv run tau3 worldgen build-v2 \
  --world data/synthetic/deposits-v2 \
  --category /tmp/term_deposits.json
uv run tau3 worldgen publish-v2 --world data/synthetic/deposits-v2

# 重读磁盘上的最终内容进行验证，而非复用阶段完成标记。
uv run tau3 worldgen validate-v2 --world data/synthetic/deposits-v2

# 显式运行 Gemini / GLM 两组 rollout。
uv run tau3 worldgen rollout-v2 \
  --world data/synthetic/deposits-v2 \
  --output data/synthetic/deposits-v2-rollouts \
  --config configs/worldgen/v2-rollout.yaml --retrieval bm25 --num-tasks 5
```

`validate-v2` 重写验证报告；若报告内容改变，已发布 manifest 的证书引用会失效，需重新 `publish-v2`。代码或产物内容变化也会使旧证书失效。旧格式的 `published` 标记不再能证明可发布；旧世界仅支持显式 `TAU3_SYNTH_ALLOW_DRAFT=1` 的开发访问。

## 产物和信任边界

| 文件 | 用途 | 是否给智能体读取 |
| --- | --- | --- |
| `spec.json` | 完整事实、规则、操作和场景契约 | 否 |
| `db.json` | 客户、动态业务表和明确标注的合成初始快照 | 通过受控工具读取客户记录 |
| `documents/*.json` | 最终公开知识库 | 是 |
| `tasks/*.json` | Tau 任务、私有目标断言、参考操作、gold | 仅将用户请求交给用户模拟器 |
| `private/articles/` | 原子主张与文档对应关系 | 否 |
| `private/reviews/` | 按最终文档哈希绑定的模型复核 | 否 |
| `validation/report.json` | 目标、证据、反例、结构多样性检查 | 否 |
| `validation/certificate.json` | 精确产物成员、内容、实现及报告哈希 | 否 |
| `progress.json`、`attempts/` | 持久预算和失败尝试 | 否 |

证书用于本地可复现性和陈旧内容检测，不是对恶意修改者的数字签名。智能体检索仅接触 `documents/`；`check_scenario_outcome` 不注册为智能体工具。

## 已实现的验证

1. **契约准入**：拒绝未知字段、标识符冲突、悬空引用、循环依赖、缺失独立目标和工作流参数错误。表达式只允许有限 AST 语法，不执行 Python `eval`。
2. **数据**：十进制金额、精度、上下界、日期、枚举、唯一邮箱、外键和客户归属；初始记录是显式合成快照，不伪称真实交易历史。
3. **操作**：精确工具别名、调用者角色、参数和记录归属；多个效果在同一临时状态中验证，通过后才提交。可声明提交后条件，例如余额减少量等于记账金额。
4. **业务义务**：`runtime` 规则阻止执行；`trajectory` 规则允许技术调用，但违规事件使任务评分失败。因此“接口允许调用”不代表“业务允许完成”。
5. **最终文档**：受控文本从事实和可执行谓词编译，布尔否定和活动日期不会被通用填充文本覆盖。自由填写的 `PolicySpec.statement` 只是候选说明，不作为受控文本事实源。
6. **证据**：展开派生事实的传递依赖、操作使用的事实、规则参数以及选品竞争者证据。gold 是在声明证明所需主张上的包含极小集合，不声称全局最少文档。
7. **任务与反例**：独立状态目标、其他客户／产品、金额越界、无操作、删步骤、顺序反转、非目标字段及中间非法改动；不把参考回放得到的数据库哈希作为唯一答案。
8. **内容评分**：拒绝和知识边界任务包含 `NL_ASSERTION`。空列表、字符串真假值、漏判、重复编号、解析错误和服务异常均不能获得语义奖励。V2 完整任务评分要求 `all`；官方域继续使用原有评分路径。
9. **发布**：验证精确成员和内容哈希，检查实现版本和报告；缺失、失败或陈旧证书均阻止加载／发布。恢复运行要求相同配置，模型调用预算跨恢复累计。

反例库能检出已覆盖的错误类型，不能证明不存在其他错误。验证器、契约解释器和文本编译器仍可能存在共同缺陷，应继续用独立回归案例和实测任务扩大覆盖。

## 无人工校对的自然语言模式

```bash
uv run tau3 worldgen build-v2 \
  --world data/synthetic/deposits-v2-natural \
  --category /tmp/term_deposits.json \
  --text-mode llm --render-model <renderer> \
  --review-model <reviewer-a> --review-model <reviewer-b> \
  --llm-config /path/to/local-request-settings.yaml \
  --max-repairs 3 --max-model-calls 150
```

文档请求配置文件直接包含模型请求参数，如 `api_base`、`api_key`、`timeout`、`max_tokens`。该文件不复制进世界，产物只保存其哈希。不要提交密钥。文档复核配置与 rollout 配置是不同入口，后者使用下面的完整模型矩阵结构。

合成 rollout 的默认网关固定为 `http://10.39.62.231:9091/v1`。服务端 `/v1/models` 返回的实际名称为 `gemini-3.5-flash` 和 `GLM-5.3-Flash`，LiteLLM 调用时使用 `openai/` 前缀。两者分别作为智能体执行同一任务集合；用户模拟器固定用 Gemini，语义评分用 GLM，不回退到 GPT。默认值位于 `src/tau3/config.py`，可复现配置见 `configs/worldgen/v2-rollout.yaml`。

`rollout-v2` 会把配置传到当前工作进程和评分路径。若直接使用 `tau3 run`，可以通过 `TAU3_SYNTH_ROLLOUT_CONFIG` 为 V2 评分选择同一个配置文件。不同世界的矩阵请使用独立进程。每次 rollout 都记录世界产物、实现和评分配置身份；不完整的旧运行不能作为通过证据复用。

每次尝试依次执行：改写 → 每位审查模型独立全文主张抽取 → 与允许主张逐条比较 → 支持性、完整性和自然度判定。至少两个不同模型标识，且至少一个与改写模型不同。每篇至少预留 5 次调用；任何一位审查者未明确通过，均不能进入发布文档集。失败意见用于下一次有限修复。

模型标识不同不保证底层模型独立，也不消除共同误判。`structural_and_model_reviewed_text` 表示通过这些检查，不表示语义绝对正确。审核未完成时返回 `INCONCLUSIVE` 和非零退出码，可使用相同参数恢复；不能通过切换到受控模式掩盖已有自然语言审核失败。

## 横向扩展

```bash
uv run tau3 worldgen scaffold-category \
  --id merchant_settlement --profile merchant --output /tmp/merchant.json
uv run tau3 worldgen expand-v2 \
  --source data/synthetic/deposits-v2 \
  --world data/synthetic/deposits-and-merchant-v2 \
  --category /tmp/merchant.json
uv run tau3 worldgen publish-v2 --world data/synthetic/deposits-and-merchant-v2
```

品类包可以声明任意数量的产品、表、规则、操作和场景，无需修改 Python 注册枚举。当前 8 个起始 profile 是例子，不是支持范围：checking、savings、term_deposit、credit_card、installment、merchant、rewards、escrow。商户、信用卡和分期例子包含业务记录与记账表；托管例子包含客户授权前置步骤。

扩展产生新目录，不改旧世界。默认要求新增结构指纹；同一模板的名称、品类标签或事实参数变化不会计为新增结构。若目的确实是参数覆盖，可显式使用 `--allow-structural-duplicates`，报告仍会计入结构重复。训练／测试按计算出的结构指纹分组，不信任作者自行填写的 family 标签。该指纹是工程近似，尚不能证明任意表达式重写或状态图同构，因此仍需要新品类的独立语义审核。

只有与经过测试的 scaffold 编译输出完全一致的品类可以免模型进行任务语义准入。任意新品类或改动后的自由文本，即使参考流程能执行，也必须提供两个模型的独立任务复核：逐场景检查请求与目标一致、输入可获得、评分断言正确、状态变化及业务规则合理。可对 `expand-v2` 追加 `--review-model` 两次及 `--llm-config`；缺失或不完整复核返回 `INCONCLUSIVE`，不能换一个来源标签绕过。

自动提出新结构：

```bash
uv run tau3 worldgen propose-category \
  --source data/synthetic/deposits-v2 \
  --output data/synthetic/category-proposals/standing-orders \
  --concept 'standing orders with customer authorization and a cancellation window' \
  --model <model> --llm-config /path/to/local-request-settings.yaml \
  --review-model <reviewer-a> --review-model <reviewer-b> \
  --max-attempts 3
```

模型输出只作为 JSON 数据解析，随后执行完整扩展验证。无新增结构、契约错误、场景失败或预算耗尽会保留为不确定候选；通过后输出候选包和已验证的新世界，仍需 `publish-v2` 明确发布本地快照。自动流程可以直接串联该命令，无需人工校对作为验证前提。

当前跨品类组合提供全局命名空间和依赖检查；外键及操作效果限制在品类自身表中。复杂跨品类资金流应建模为包含所需多张表的复合品类。尚不能把声明的品类依赖误认为已经支持任意跨品类联动。

## 难度复现与实验门禁

结构验证通过后，报告中的论文难度仍是 `INCONCLUSIVE`。`qualify-v2` 接受逐一配对的 Tau 结果，检查：

- 合成结果的产物和实现身份，与当前世界一致。
- 两侧模型、用户模拟器、检索配置、种子、步数预算、重复次数等一致。
- 每个声明任务／trial 都有完整轨迹和评分，不能排除服务失败或截断后再算一个好看的通过率。
- 每个所需奖励分量确实被评估；任务内容与当前合成／官方源一致。
- 至少两个智能体模型与三种检索配置的完整组合，含 `golden_retrieval`、`no_knowledge`。
- 每侧至少 50 个任务；按任务计算所有重复均成功的比例，保守的 Wilson 区间差必须完全落在默认 ±0.1 等价区间内。样本不足或不确定性过大为 `INCONCLUSIVE`。

```bash
uv run tau3 worldgen qualify-v2 --world <world> \
  --synthetic <model-a-gold-synthetic-results> --control <model-a-gold-official-results> \
  --synthetic <model-a-search-synthetic-results> --control <model-a-search-official-results> \
  --synthetic <model-a-none-synthetic-results> --control <model-a-none-official-results> \
  --synthetic <model-b-gold-synthetic-results> --control <model-b-gold-official-results> \
  --synthetic <model-b-search-synthetic-results> --control <model-b-search-official-results> \
  --synthetic <model-b-none-synthetic-results> --control <model-b-none-official-results>
```

结果身份由 runner 自动写入，旧结果不能凭路径相同复用。`paper_reproduction` 发布还要求自然语言审核及有效行为证据；采样实验可用显式 draft 开发加载运行。即使通过此比较，也只能说明所测条件下的行为接近，不能由通过率推导整个任务分布等同。结构、文档自然度、gold 长度、任务类型比例和外部真实业务覆盖仍应单独分析。

## 验证与持续运行

```bash
uv run pytest tests/test_domains/test_banking_knowledge/test_worldgen*.py -q
uv run python scripts/validate_worldgen_v2.py
```

规模脚本构建并发布 4／12／24 品类的受控文本快照，分别报告文档数、任务数、结构数和拒绝的反例数。这是容量和隔离检查；复制同一 profile 不会凭空增加业务多样性。正式大批量生产应运行候选提出 → 新结构准入 → 文档双审 → 完整验证 → 目标用途实验 → 发布，并持续监控拒绝原因、模型预算和结构重复率。
