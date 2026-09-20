# v03 驱动的原生银行合成轮次 r10

配置：`configs/synthesis/targeted-native-v03-r10.json`。本轮与 r9 分离，不改写旧产物。
实现沿用 native v2 的真实上下文捕获、请求日志、固定四 seed、全局请求预算和 256 并发池。
`curriculum=v03_r10` 强制 `clean_only=true`。

## 冻结配额

| 主族 | small | full |
|---|---:|---:|
| credit_limit | 100 | 750 |
| transaction_disputes | 80 | 600 |
| replacement_closure | 60 | 450 |
| accounts_funds | 48 | 360 |
| debit_security | 40 | 300 |
| optimization | 32 | 240 |
| handoff | 20 | 150 |
| escalation_boundary | 20 | 150 |

试跑为 20 题，独立验证为 200 题。难度按参考动作数分配为
15/25/25/20/15%，来源为 current/replay/explore 70/20/10%。正式额度族中
250 批准、300 拒绝、100 提交前停止、100 多卡混合。拒绝覆盖七种有明确政策条件的原因；
金额超过上限进入提交前停止，不为了覆盖枚举而制造不合规的超额申请。

`v03_planning.py` 在主族内为少数操作保留明确配方，防止主族数量达标但具体工具从未出现。
每个 slot 冻结实际参考操作名，内容重建也必须保留这些操作。覆盖按 task 计数，不按调用次数累计。
覆盖契约从安装的 97 个官方 task 提取精确操作名和当前签名，不把官方客户状态或答案复制进候选。
当前统计包含公开和用户侧操作，因此不能直接与旧报告的“归一化 agent 操作数”比较。

高利用率反转对共享业务实体与随机参数，只改变初始余额；自动检查恰好一个源字段变化，
两个成员进入同一划分。验证结构组在采样前确定；只有一个可执行结构的分支不会假借新客户 ID 声称独立验证。

## 任务和轨迹验收

- 内容准入仍需静态正例、独立业务终态、反例、两模型公开盲解与文档审核。
- CLI oracle 独立从账户、申请历史、支付历史与公开规则推导批准/拒绝，检查全部必需查询和决策顺序。
- 反例包括漏 unlock、错误工具后缀、错误 selector、错误 JSON 类型、缺参、多参、隐藏工具直调、漏写、额外写以及将拒绝改成批准。
- 无参工具的 `{}` 是合法参数，不生成“缺参”伪反例。附加合法查询可以通过。
- 新静态证明保存 `expected_db_diff`；成对证明保存两个业务指纹和改变的字段。
- 只有正常成功且质量审核通过、整个过程没有工具或协议错误的轨迹进入主集。
- 成功恢复轨迹单独写 `sft_recovery.jsonl`，错误 assistant 轮（包括其 reasoning）全部 mask。
- 教师原始 reasoning 仍须通过正确性、可见历史依据和无私有信息泄漏审核；不补造缺失 reasoning。

每个阶段导出：`sft_clean.jsonl`、与其对应的 `sft.jsonl`、
`general_agent_reasoning.jsonl`、`sft_recovery.jsonl`、`training_manifest.json`、
`operation_coverage.json`、`pair_checks.json`、`v03_metrics.json`、`hard_pool.json`。
训练 manifest 包含内容 hash、原始 capture/质量证明绑定、token mask 检查和 task 内轨迹等权权重。
恢复集不参与主集产出率。未决采样仍保留在分母中。

## 外部评测门

先完成 pilot、small 400 和 validation 200，再由外部训练并提交 receipt。
full 受现有 `accept-evaluation` 入口控制。r10 增加 task_050–054、task_051、协议错误下降、
幻觉工具下降、第一次 discoverable 调用格式成功、错误实体写入不增加等门槛。
严格比较模式保留 reasoning 设置和网关地址的 hash；不同推理设置不能被过滤后伪装为可比。
既定提升门槛即使遇到高基线而不可达，也不会自动放宽。

## 离线预检

```bash
LOGURU_LEVEL=ERROR .venv/bin/python scripts/check_targeted_native_v03.py \
  --config configs/synthesis/targeted-native-v03-r10.json \
  --output data/synthesis/targeted-native-v03-r10-preflight \
  --static --full
```

此命令不调用 LLM。`probe-plan.json` 使用明确的探针画像，不能作为生产证据或有效任务准入。
`--static` 回放 pilot 的正反例；全批次输出是编译及覆盖检查，不等于 3000 个已准入有效 task。
少数业务分支另有无网络的回归测试。

## 已确认的隔离环境修复

官方默认 `approve_credit_limit_increase_5847` 在 submit 后尝试插入相同 request ID；
重复插入被拒绝，但额度已改变、申请仍为 PENDING。本轮经用户确认采用
`runtime_revision=native_cli_approval_v1`，只在任务初始 `task_config.native_runtime.revision`
携带该标记时更新匹配的 PENDING 申请，保留提交字段并写入批准字段。

批准前检查申请所属用户、账户、金额、状态和已有业务资格条件；不匹配时返回错误，
不改变卡额度或申请记录。未标记的官方任务保持原行为。修复、配置、初始状态均进入版本证据。
回归测试覆盖批准与混合流程、未提交、错金额/用户/账户/状态、资格失败及官方默认行为。

完成预检后，生产监督器入口为：

```bash
LOGURU_LEVEL=ERROR .venv/bin/python scripts/run_targeted_native.py \
  --round-dir data/synthesis/targeted-native-v03-r10 \
  --config configs/synthesis/targeted-native-v03-r10.json \
  --report data/calibrations/docs/v03.md \
  --additional-report docs/tau3-banking-failure-analysis.md \
  --round-id targeted-native-v03-r10 \
  --base-model 'dots3-post-sft-v03-report-checkpoint-unspecified'
```

`parent_round` 从配置绑定 r9 的 plan hash。LLM 画像抽取支持多个报告，每条原文引用仍须落在一个真实源文件内。
外部训练与评测未提供前，监督器只能到 small/validation 交付点，不声称 full 已完成或官方指标已改善。

## 2026-09-19 验证记录

- native 与 V2 审核/streaming 兼容测试：183 通过；后续协议分类检查 9 通过、额度检查 10 通过（与前者存在重叠，不相加）。
- 本次涉及文件的只读 Ruff 检查通过。
- `make test`：277 通过、17 失败、1 xfailed；真实模型/API 凭证相关测试未通过。
- `make test-knowledge` 首次全量：946 通过、61 失败、18 setup errors、14 skipped。
  native 初始化测试遗留并发环境变量导致的 V2 兼容失败已修复并单独回归通过；
  外部检索 API/依赖及旧 adapter 固定 hash 的失败仍保留，不能声称全量 knowledge 已通过。
- 全仓库只读 Ruff 仍有 55 项本次 native 修改范围外的问题，未自动格式化或改写无关文件。

运行状态写在 `data/synthesis/targeted-native-v03-r10/implementation-status.json`。
已确认隔离修复；实际阶段、进程和完成状态以本轮 `supervisor.json` 及阶段证据为准。

2026-09-19 隔离修复验证：批准专项回归 11 通过；pilot 静态正反例 20/20 通过；
400/200/3000 候选的离线编译、操作覆盖与配对检查均通过。后台监督器已启动，
实时状态见 `supervisor.json`；不能把离线候选数称为教师完成数。
报告涉及多个 dots3 版本而未绑定精确 checkpoint，因此画像明确记录 checkpoint 未指定；
外部训练 receipt 仍须绑定实际 checkpoint。

隔离修复后的完整 native/V2 兼容回归：194 通过（169.54 秒）；只读 Ruff 涉及文件全部通过。
