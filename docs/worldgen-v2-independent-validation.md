# V2 独立验证与受控扩展准入

本阶段保留 V2 为可靠种子生成的基础版本。基础发布证书证明结构、执行和已声明文本审核条件，不代表大规模真实语料生产线验收。扩展准入单独检查，不替代论文行为对照。

## 2026-09-12 实测结论

该轮实现对应的审计目录为 [`worldgen-v2-guarded-high`](../data/synthetic/worldgen-v2-guarded-high/)。使用指定网关的 Gemini／GLM，实际请求均携带 `reasoning_effort=high`。本轮受控验收为 **INCONCLUSIVE**，完整验证和扩产仍未解锁。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 独立盲解 | 12/12 PASS | 六个代表任务，两个模型分别解题、交叉审核 |
| 已保存盲解证据重放 | 12/12 PASS | 重建公开输入，复查原始响应、冻结结果、计算与状态 |
| 真实模型评分校准 | 169 PASS，1 INCONCLUSIVE | 八种业务结构的 85 个固定正反例 × 两个评分模型 |
| 在线回放 | 24/24 成功，reward 均为 1 | 六个任务 × 两个模型 × full_kb／bm25 |
| 离线回归 | 276 passed | worldgen、评估轨迹、环境和 runner 相关测试；Ruff 检查通过 |

唯一未决校准为 GLM 的 `denial_near_eligibility_date`：响应正确指出日期差一天，但两个判定的索引均为 0，缺少断言 1。原生 JSON 模式只保证语法，不能保证完整的判定契约；严格校验按协议异常拒绝该响应，没有推断或人工改写索引。当前组合评分器在有效判定中未观察到误放行或误拒绝，这不包括未决项，也不代表总体错误率为零。原始响应及定位结果见审计目录的 `calibration-schema-diagnostic.json`。

此前 `worldgen-v2-json-high` 实验曾捕获 Gemini 将“操作成功、最终答复为空”误判为成功。现已增加最终答复的确定性否决，并用该次真实模型原始响应、完全相同的请求离线重放，确认 reward 从 1 变为 0；记录在 `silence-replay-regression.json`。这是回归验证，不计作新的在线模型判定；此前失败实验仍完整保留。本轮校准没有触发这项否决，报告的 `deterministic_vetoes` 为 0。

在线样本覆盖 service、selection、denial、ordering、grounding 五类任务，以及预约取消流程；**尚未完成全部既有任务、品类和业务结构的在线验证**。由于校准门槛未过，计划中的 488 格完整范围未执行，本阶段也未新增自然语言文档或品类。刷新了原有 5 品类／21 任务、24 品类／120 任务及单任务自然语言样本的副本，刷新证书本身不等于取得扩产资格。

下一步首先补齐评分协议异常的有限自动修复：只把格式／索引覆盖错误反馈给同一个评分者，不提供预期标签，不重试有效但错误的语义判定；保留首轮失败并把修复调用计入固定预算，修复失败仍为 INCONCLUSIVE。该轮结束时修复尚未实现；现已补齐，见下节。之后需要以新审计目录重新验证当前实现，门槛通过才运行完整既有任务矩阵，再决定是否扩大自然语言语料。新增业务仍须同时提供验证能力、确定性检查和带固定标签的评分反例。


## 2026-09-13 评分协议修复

评分器现在对无效 JSON、错误字段类型及断言索引覆盖错误，向同一个模型最多追加一次协议修复请求。请求保留原有评分输入，只增加原始无效响应、协议错误和所需索引；不加入预期标签、私有参考操作或目标状态。合法的语义判定无论正确与否都不重试，服务错误、预算耗尽和非正常结束也不触发协议修复。

每次评分的 `reward.info.nl.protocol`（独立语义评分为 `reward.info.protocol`）保留两次尝试的响应、格式错误、修复反馈和最终状态。校准审计中的两次请求分别占用调用预算；恢复运行复用精确请求的原始响应，不删除首轮异常。协议再次失败仍为 `INCONCLUSIVE`。校准报告的 `protocol` 统计首次有效率、修复尝试数／成功率和最终判定正确率；最终正确率的分母包含未决项。修复尝试数包含被预算拦截的尝试，实际预留模型调用以 `calls/` 记录为准。

本轮采用新的 `data/synthetic/worldgen-v2-protocol-high/` 审计目录，旧实验保留。副本从上一轮种子迁移并重新验证，不重写公开文档或业务数据。完整任务矩阵仍须等待本轮 smoke 全部通过。

针对性测试 54 项、相关完整离线回归 285 项通过。`protocol-repair-probe.json` 记录了定向网关实验：重放上一轮真实的重复索引响应，再发起一次真实 GLM 修复请求，得到索引 0／1 的完整判定，且仍指出资格日期错误。该实验是“历史失败响应 + 新修复调用”的回归验证，单独记录，不纳入全新校准的通过率。

本轮最终实测：独立盲解 12/12 PASS，保存证据重放 12/12 PASS；评分校准 **170/170 PASS**，首次响应有效率 100%，无需修复，因此本轮校准的修复成功率为不适用，而不是 100%。有效判定没有观察到误放行、误拒绝或确定性否决。上述定向实验另有一次真实修复成功。

在线矩阵执行了全部 24 格，其中 **22 格成功、2 格 INCONCLUSIVE**，均为 GLM 在 BM25 条件下触及固定的 40 步上限：`task_category_000_selection` 发起 15 次知识检索后尚未开户；`task_category_000_ordering` 已收到 CLOSED 工具结果并答复，但在正常终止前触及上限。原始轨迹未替换，步数上限未调整，也未将第二条人工改判为通过。详见 `online-limit-diagnostic.json`。

因此，原先的评分协议阻塞已解决，但整体 smoke 仍为 **INCONCLUSIVE**，完整 488 格范围和扩产保持锁定。下一步需要解决检索重复、正常收尾及步数预算的适配问题，以预先声明的新实验重新验收；不能通过提高上限后覆盖旧失败来宣称本轮通过。种子文档、任务与业务数据内容均保持不变，本轮只刷新发布清单、迁移记录和证书。

## 验证边界

`blind-v2` 只向模型提供用户请求、公开政策、全部公开文档、当前日期，以及通过公开查询获得的该客户身份和记录。它不序列化私有 Task、参考操作、目标字段、评分断言或 gold 文档列表；也不使用 `golden_retrieval`。

两个模型分别提交答案、原文引用、操作计划、完整的变更后记录和计算过程。原始响应先写盘冻结，再交另一个模型只对照公开材料审核。随后才执行计划并核对私有目标，同时拒绝非目标记录、非目标字段、身份或种子历史的变更，也检查已撤销的中间副作用。独立的 Decimal／日期检查器不调用生成器的表达式解释器。准入复查会重新构造公开输入，检查原始响应与冻结结果一致，重新核验引用、计算及执行结果。

这是模型输入层面的隔离，不是操作系统沙箱。操作回放仍使用同一个业务运行时；模型之间也可能共同误解文档，因此不能把“双模型通过”当成数学证明或现实金融真实性保证。公开资料本身是否真实，仍超出合成虚构银行的保证范围。

`calibrate-v2` 使用固定预期标签的正反例，不让模型自行决定正确标签。它通过实际的 `evaluate_simulation` 同时验证状态和语义评分，两个模型分别担任真实评分者。样例按业务结构指纹选取，避免只覆盖同一任务类型的第一种流程；预约取消和多表记账等结构各自需要正反例。测试包括错误金额／产品／客户、顺序和授权错误、正确状态下的虚假回复、空回复、矛盾回复、评分指令注入，以及分两次处理相同总额的合法替代流程。固定反例通过不等于总体误判率为零。

全部 V2 任务现在都包含最终沟通一致性断言。最终助手消息必须是有文字内容的客户可见答复；只有工具调用、空白或标点不能取得沟通分。模型仍会被调用并保留原始判定，确定性规则可否决错误的通过判定，校准报告单独统计 `deterministic_vetoes`。反例准入衡量组合评分器是否正确，不等于模型自身零误判。评分输入保留工具调用参数、实际结果和错误标记。服务异常、非正常结束（包括 `length` 截断）、响应格式不合法等标记 `INCONCLUSIVE`，不能算作正确识别反例。

评分器还读取任务关联的公开证据文档、世界日期和客户初始状态，以核对政策、资格日期和数值事实。评分阶段允许使用证据文档列表；独立解题阶段仍读取完整公开库，完全不接收该列表、参考操作或目标状态。反例包含正确操作后仅差 0.01 的金额、错误产品标识，以及差一天的最早资格日期。

## 命令

继续使用指定的 Gemini／GLM 网关。本轮验证使用 `configs/worldgen/v2-verification-high.yaml`：请求输出上限 65,536，单次请求超时 180 秒，单条仿真超时 900 秒，推理预算 high，无隐式重试。通过 `extra_body.reasoning_effort` 传递，已抓取实际 HTTP 请求确认两个模型均携带 `high`；普通顶层 LiteLLM 参数在这些自定义模型名下可能被 `drop_params` 丢弃。请求上限不等于服务端实际生效上限：网关修复前，复杂 Gemini 审核曾观察到 8,192 个推理 token 后 `finish_reason=length` 且无正文，相关检查保持 `INCONCLUSIVE`。用户修复并重启网关后，同一复杂请求已正常返回 17,899 个 completion token（其中 17,859 个为推理 token）；此前截断记录仍保留，不能当成成功调用。原有 `v2-rollout.yaml` 保留为基础配置。以下路径是示例，需替换为当前已重新验证的世界。

```bash
uv run tau3 worldgen blind-v2 --world WORLD --output AUDIT/blind --max-calls 1000 --config configs/worldgen/v2-verification-high.yaml
uv run tau3 worldgen calibrate-v2 --world WORLD --output AUDIT/calibration --max-calls 400 --config configs/worldgen/v2-verification-high.yaml
uv run tau3 worldgen matrix-v2 --world WORLD --output AUDIT/online --max-rollouts 500 --config configs/worldgen/v2-verification-high.yaml
uv run tau3 worldgen readiness-v2 --world WORLD --output AUDIT/readiness.json \
  --blind AUDIT/blind --calibration AUDIT/calibration --online AUDIT/online --config configs/worldgen/v2-verification-high.yaml
```

盲解和在线矩阵可重复指定 `--task-id` 运行固定子集。在线矩阵默认包含世界全部任务，以 `full_kb` 和 `bm25` 分别运行两个模型，按任务类型、业务结构、品类报告覆盖率。矩阵 `PASS` 仅表示完整、有效地执行，不表示每条轨迹都成功。每个任务至少一个 `full_kb` 成功实例，以及两个经过核验的盲解结果，是扩展准入的额外要求。

审核调用在发出前预留预算，保留精确输入、原始响应及失败类型。不同产物、实现、配置不得复用旧审核。可在文件指纹完全相同的本地镜像上恢复同一审核，以减少共享文件系统的重复读取；原始响应仍保存在审计目录中，迁移必须在没有在途模型请求的检查点进行。失败或中断的请求不会被当作成功缓存。需要重新实验时使用新的审计目录；不得删除失败证据来装成首次通过。

结构化审核和评分请求通过 `extra_body.response_format={"type":"json_object"}` 启用网关原生 JSON 模式，普通对话与工具调用保持原有输出接口；本地校验仍严格拒绝无效响应。

每个盲解最多尝试四次，每行解题与审核合计最多八次调用，另受整个实验的总调用预算约束。审核格式错误会将公开格式反馈发送给审核者，并保持候选解不变；语义审核指出的问题才交给解题者修正。修正反馈只来自 JSON 格式、公开引用、独立计算和公开交叉审核；私有目标核对发生在修正结束、结果冻结之后，绝不作为修正提示。JSON 可使用单个完整 Markdown 围栏，但重复对象键、非有限常量及多个对象会被拒绝；不会从杂文或多个对象中挑选一个看似通过的结果。

用户端的 `list_granted_tools` 只返回本次环境实际授予的工具名与参数契约。公开政策还要求向客户询问缺失的客户指定标识，包括新建记录所需标识。交接政策要求助手明确告知精确名称和参数，避免用户模拟器把产品标识或自然语言描述当成能力名称。未授予的工具仍不可调用，不同仿真间不共享授权。

`expand-v2` 和 `propose-category` 默认要求 `--readiness` 指向当前通过的准入报告。使用自定义验证配置时还需传 `--verification-config configs/worldgen/v2-verification-high.yaml`，与准入证据的配置指纹一致。`--foundation-only` 是显式的基础开发范围，产物仍不具有扩产资格。本阶段不使用这个选项绕过失败后继续扩产。

新品类可声明 `validation_capabilities`。目前覆盖公开证据、状态投影、十进制计算、日历日期和角色权限；未实现的验证能力会使盲解准入保持 `INCONCLUSIVE`。新增复杂业务必须扩展检查器及有预期标签的正反例，不能仅注册一个能力名称来取得可靠性。

## 固定实测流程

```bash
uv run python scripts/validate_worldgen_v2_independent.py --stage prepare --output RUN
uv run python scripts/validate_worldgen_v2_independent.py --stage smoke --output RUN --config configs/worldgen/v2-verification-high.yaml
uv run python scripts/validate_worldgen_v2_independent.py --stage full --output RUN --config configs/worldgen/v2-verification-high.yaml
```

`prepare` 复制既有快照，更新评分元数据并重新验证；不新增文档、品类或业务场景。原始快照保留。

`smoke` 选取五类基础任务及预约取消流程，共六个任务。执行双模型盲解、真实评分正反例和 24 次在线 rollout。所有门槛满足后，`full` 才运行 120 个既有任务及自然语言样本；连同 smoke 中取消流程的四格结果，覆盖计划中的 488 格在线实验。完整阶段的每个盲解分片在首个必需检查未通过时停止，保留未覆盖任务为 `INCONCLUSIVE`，任一分片不完整都会阻止后续阶段；不得把“未启动完整矩阵”写成“完整验证通过”。

并行启动的三个 smoke 工作进程结束后，可在上述 smoke 命令加 `--summarize-only`，只聚合已保存报告，不触发新的模型调用或重试。

可用 `--local-worlds RUN/local-worlds.json` 为验证驱动指定已准备的本地镜像；JSON 映射键为 `world`、`scale`、`natural`。驱动会核对原目录与镜像的完整产物指纹和证书，结果仍写入 `RUN`。省略该选项时直接读取原目录。

## 后续检索与 task 准入改进（验证进行中）

`configs/worldgen/v2-verification-catalog-high.yaml` 为新实验固定 80 步，旧配置
仍保留 40 步；正常终止、状态和语义要求不变。公开目录
`KB_list_documents(query, offset)` 只检索文档 ID／标题，支持分页；
`KB_read_documents(document_ids)` 每次最多读取十份完整公开文档。它们只在
有检索工具的环境暴露，不读取私有目标、参考操作或 gold 文档列表。

独立计算器支持明确的字面量数值相等记法，例如 `5=0`，仍按独立 Decimal
比较得到 false；其他非法算式会提供包含原表达式的公开修复反馈。默认
盲解引用只提交公开来源 ID，程序附上该来源的完整原文，供交叉审核和冻结
证据使用；不要求模型复制引文或手数行号。可选的 1-based 行范围会严格
核对范围及对应原文；显式错误引文不会被替换为正确事实。保存证据复查会
从原始模型响应重新执行同样的确定性提取，不能仅信任已加工的引用字段。
修复提示使用原始候选形式，不把程序附上的引文再交给模型改写。

`worldgen-v2-catalog-high`、`worldgen-v2-ready-high` 和
`worldgen-v2-span-high` 的中断记录保留了原始算式／引文问题和已完成结果，
不作为新实现的通过证据。当前 `worldgen-v2-source-high` 实验先完成盲解，
通过后才启动评分校准与在线矩阵，再按原有门槛进入完整范围。

新增 `tau3 synthesize world-tasks` 及 `scripts/run_world_task_synthesis.py`，
用于已通过准入的 V2 种子。超过 20 个任务的批次必须提供同配置下已发布的
20 任务试运行。每个新任务重新经过公开表达核对、双模型独立解题和两条
真实 BM25 回放；最多三个候选，服务异常及预算耗尽不会变成通过。详细
范围与命令见 [task 合成说明](task-synthesis.md#v2-synthetic-seed-worlds)。
这些入口的存在不代表当前种子或试运行已经验收通过。
