# 失败报告驱动的 V2 task 合成

`tau3 synthesize targeted` 为每轮外部训练提供定向业务任务及教师 SFT 数据。
官方报告用于选择能力目标，任务运行于虚构的 `banking_synth` 世界。
它不修改官方政策、业务工具、评分或 base model，也不启动训练作业。

默认配置是 `configs/synthesis/targeted-v1.yaml`：20 个试跑任务，3000 个正式
训练任务，每题固定四次 GLM 教师采样。所有通过环境评分及 Gemini 质量审核的
成功轨迹均保留。四次失败的有效任务进入难题池，不通过简单任务补齐产量。

## 运行

可以使用阶段监督脚本运行完整流程；任一阶段未决即停止，不会跳过试跑启动正式批次：

```bash
python scripts/run_targeted_synthesis.py \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml \
  --report docs/tau3-banking-failure-analysis.md \
  --run data/simulations/tau3_banking_dots3_max \
  --round-id dots3-round-000 --base-model openai/d3note
```

`run-state.json` 保存当前阶段、子进程和心跳。添加 `--resume` 恢复相同实验。
默认 `workers: 128`、`llm_concurrency: 128`。归因批次使用线程安全的直接
请求并行执行；世界任务使用独立进程。所有模型角色的物理请求共享每轮
128 槽位的限流池。世界进程将供应商异常转为普通失败记录，避免无法反序列化的
异常导致整个进程池损坏。
`run-state.json` 的 `llm` 字段以及 `llm_pool/status.json` 记录当前请求数、
实际峰值与上限。归因批次数或试跑任务数少于 128 时，实际并发低于上限；
报告解析和配额规划本身只有一个有依赖关系的请求，不会重复调用来凑满并发。

需要 knowledge、gym 和 dev 依赖。模型端点为
`http://10.39.62.231:9091/v1`，教师和规划者为 `openai/GLM-5.3-Flash`，
用户模拟器和质量审核者为 `openai/gemini-3.5-flash`；两个模型分别参与盲解
和评分校准。配置不接受持久化的真实 API key；需要认证时使用 provider 环境变量。

```bash
tau3 synthesize targeted analyze \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml \
  --report docs/tau3-banking-failure-analysis.md \
  --run data/simulations/tau3_banking_dots3_max \
  --round-id dots3-round-000 --base-model openai/d3note

tau3 synthesize targeted plan \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml

tau3 synthesize targeted pilot \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml

tau3 synthesize targeted generate \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml

tau3 synthesize targeted collect \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml

tau3 synthesize targeted export \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml

tau3 synthesize targeted report \
  --round-dir data/synthetic/targeted/round-000 \
  --config configs/synthesis/targeted-v1.yaml
```

`--run` 可省略；只有报告时缺失统计保持未知。下一轮 `plan` 可传
`--previous-plan PREVIOUS_ROUND/plan.json`，历史权重参与 20% 回放配额。
同一阶段再次运行须添加 `--resume`，配置和实现身份必须相同。更改实现、预算、
模型或输入后使用新轮次目录，旧失败证据不会被改写成成功。

提供原始运行时，分析器先按 task/trial 保存实际调用、预期动作、参数差异与
检索重复，再分批交给 GLM 归因。每项归因引用程序生成的稳定证据 ID，程序校验
ID 所属 task、原始 trial、字段内容哈希和源文件哈希，且至少引用一条失败 trial。
只有这些带证据的归因参与 task 级权重。程序标注的现象假设不会直接变成根因
或配额；缺少足够归因时回退到报告的定性优先级。
归因输入按 `analysis_batch_bytes` 分批，跨批保留原始 task/trial 索引；
单条证据超过预算时显式停止，不静默截去动作或参数。
归因请求单独使用 `attribution_timeout: 1800` 秒，其他模型角色和轨迹预算不变。
归因的传输、结构或证据校验失败最多触发一次显式恢复：关闭该请求的原生 JSON
模式，仍要求完整 JSON 和相同证据 ID 校验。原响应不覆盖，恢复请求使用独立
指纹并计入预算，结果保存在 `attribution-recovery/`；成功批次复用，不重新采样。

`pilot` 串联准入、80 个教师采样槽位及试跑导出；正式 `generate` 只在试跑
完整、至少产生一条 SFT 样本并且预算预测通过后启动。试跑不要求每个难题都
刷出成功教师轨迹。阶段未完成返回退出码 2，正式任务内容无效时最多尝试三个
同配额候选；服务、预算或验证未决不会触发降低难度或隐式换 seed。

## 合成与验证

七类编译器覆盖发现/枚举、政策推导、用户交接、写入克制、业务原因升级、
签名遵循和信息边界。当前实例是多账户/卡下的 case review：记录政策计算出的
金额估计、责任档位、交付等级和处理路由，不代表真实银行争议处理法律，也不
把记录估计伪装为支付。探索任务组合两种服务机制。

业务状态及独立目标由程序生成，GLM 改写公开请求，Gemini 核对意图等价。
同形状参数对只改变报告日期，跨越政策边界；业务指纹排除世界 ID 重命名。
冻结任务配额中 70% 来自本轮缺口、20% 历史回放、10% 探索。
难度按编译后的参考动作数分配，另记录业务写入数量，教师表现不修改配额。
LLM 提出的新生成器进入 backlog；未实现的任意参数或新业务代码不自动执行。

查询采用带投影、过滤及分页的可发现只读操作，按客户和产品限制返回值。
定向世界关闭全量记录快捷工具和文档目录；教师使用 `bm25_grep`。盲解检查器
通过相同的公开查询重建可见记录，再提供完整公开文档进行独立求解。
查询不会更改业务状态；合法的不同查询顺序、写入顺序不因参考轨迹不同被拒绝。

每个世界经过确定性正反例、独立 Decimal/日期核对、双模型 category 复核、
双模型公开盲解、真实评分校准和 V2 readiness。校准包含错金额/档位/路由、
额外参数、漏对象、重复写入、错误用户执行、虚假支付回复、空回复、评分指令
注入和合法替代顺序。定向入口使用 `targeted_training` readiness，只接受完整的
独立盲解及真实校准证据，不按额外在线试解的成功率筛选任务；旧 V2 扩展入口
继续要求在线验收。教师成功率由冻结的四次采样测量。
受控文档从契约编译，当前不自动开展自由文档扩写。

## 产物、预算与恢复

- `profile.json`、`report-source.json`：源报告、观察与根因假设、评分疑点。
- `plan.json`：冻结配额、任务槽位、父计划身份和扩展待办。
- `pilot/`、`formal/`：独立产物树；每个槽位绑定唯一世界及其验证证据。
- `formal/slots/*/teacher/{0,1,2,3}/`：原始运行、可见上下文、质量审核及训练样本。
- `formal/sft.jsonl`：仅正式训练成功轨迹；标准 messages/tools 与 assistant-only
  `loss_mask`。reasoning 元数据、私有用户工具消息和评分答案不进入 agent 上下文。
- `formal/report.json`、`hard_pool.json`、`incomplete_pool.json`：产量和完整性分开统计。
- `physical_calls/`、`budget.json`、`budget-forecast.json`：所有实际请求、失败预留、
  token 统计及正式运行前的试跑预测。原始响应可包含长工具上下文，应预留磁盘空间。

预算覆盖教师、用户、评分、生成和盲解请求；12000 仅为正式教师轨迹数。
加上试跑的 80 次教师轨迹，默认总上限为 12080 次。跨实验恢复时，历史中断的
在线诊断或采样预留仍累计计费，并明确记录在新版本的预算中。
物理请求边界禁用隐式重试，失败也消耗调用额度。请求前按输入字节数加输出上限
保守预留 token，收到可信 usage 后结算；失败或缺失 usage 保留原预留。
显式传输重试上限由 `settings.audit_transport_attempts` 控制，当前配置为 3。
审核请求对连接、限流、服务端错误及已识别的 Gemini 后端 `brpc [E401]user auth failed`
执行有限重试；其他在线请求只对该已确认的间歇性后端认证错误恢复。
普通 API key 错误不会重试。每次物理尝试都单独计费，保持模型、上下文及 seed
不变，不创建额外教师采样槽位。修改重试配置必须建立新实验版本。
SFT 审核缺少字段或字段类型错误时，允许一次输出协议修复，保留首次响应和
修复响应的证据绑定；合法的负面审核结论不重试。协议修复只使用已经捕获的
同一条轨迹，不重新采样教师，也不把评分意见写入训练上下文。
正式预算预测使用试跑开销乘规模和 1.25 安全系数，预测通过并不免除运行时硬上限。

同一轮次只有一个阶段写入者，世界在独立子进程中运行。已经启动但没有完成证据
的采样不会重跑；已完成的捕获能继续未启动的质量审核。修改审核结论不能绕过
导出时对原始审核响应、native run、capture、样本和世界身份的核对。

`teacher_pass@4` 和 `sft_yield@4` 的分母始终包含全部计划 task；另外报告四次
完整率、完整任务的 0–4 成功直方图以及按能力/难度拆分结果。不完整轨迹不会
伪装成普通失败，也不能通过删除它们抬高通过率。

本配置全部任务用于训练，没有合成 holdout。官方评测已经参与适应性数据选择，
因此属于开发反馈；迁移收益需要外部 SFT 后重新实测，合成验证不能代替这一结论。

离线测试：

```bash
pytest tests/test_domains/test_banking_knowledge/test_targeted_synthesis.py -q
```

### 持续运行中的增量导出

`scripts/run_targeted_continuous.py` 将导出放在独立子进程，主调度不等待导出。
导出子进程使用 `incremental_export.py`，默认以 8 个独立进程校验任务，并将结果
原子保存到 `formal/export-shards/`。每次复用分片仍检查所有已绑定源文件的内容
哈希和文件集合；只有输入完全一致才省去重复准入验证、盲解重放和样本构建。
不能仅根据文件时间戳复用共享文件系统上的证据。

导出优先处理尚未出现在训练文件中的任务，每 30 秒左右在有新校验结果时合并
一个完整 JSONL 检查点。`export.pending.json` 是发布事务的恢复记录，可处理
JSONL 已替换但 `export.json` 尚未更新的中断。按 task、trial、seed 去重，冲突
样本或证据校验失败使导出报错；不会将失败任务强行转换为成功样本。

进度见 `formal/incremental-export-progress.json`。全部采样槽位完成后，导出还需
跳过缓存重新执行完整准入和轨迹验证；`full-export-audit-required.json` 请求该
最终检查。最终完整状态仍以任务数、四次采样覆盖、质量结果和导出绑定共同判定。

对运行中的冻结教师环境，可以在轮次目录的 `export-runtime/` 保存单独版本的
`targeted_incremental_export.py`，由导出子进程加载。导出代码版本、测试和基准
结果记录在 `operational-revisions/`，不修改教师模型、seed 或累计预算。
