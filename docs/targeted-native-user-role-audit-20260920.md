# 用户端消息边界与角色错位核查（2026-09-20）

结论：客户端未发现 Agent 私有上下文串入用户端；已经确认 Gemini 用户模拟器生成客服式回复，以及原质量审核误放行。两类问题需要区分，API role 翻转本身是正确机制。

## 全量请求证据

对 repair8 全部 890 条正式 SFT，沿 adopted-capture.json 找到真实原始采样目录，再逐个关联 response journal 与 physical_calls。共核对 3609 次用户模型请求（包含用户工具轮，因此多于 SFT 中 2635 条可见用户文本）。每次检查模型、物理响应与捕获绑定、唯一客户 system prompt、原始 user scenario、历史角色转换、用户工具范围。

全部通过：真实客服公开消息在用户 API 中为 user；模拟客户自己的消息在用户 API 中为 assistant；用户工具响应保持 tool。未见真实 Agent 历史被当成客户自己的 assistant 历史，也未见 Agent system prompt、私有工具执行历史或 reasoning 被拼入客户上下文。这个结论针对客户端记录，不能替代远端网关内部日志。

例如正式第 5 行：第一次请求仅有客户 system prompt 和 user=`Hi! How can I help you today?`。Gemini 原始响应却为 `I'd be happy to help ... verifying your identity ...`，随后 Agent 明确纠正角色。原质审 user_compliant=true。第 30、43 行存在同类已确认问题。

## 分布和当前运行

原词法筛查命中 107/890 行（12.02%）、108 条用户消息；其中 106 条位于第一次用户回复。该比例是候选比例，不是全量语义确诊比例。旧 890 条均可直接使用的假设需要撤回并复审。

当前 expand3k 运行的时点检查覆盖 409 次用户端请求，其中 407 次已完成，均使用 Gemini。客户 system prompt 没有串入 Agent prompt，但筛出 24 条客服式候选；直接核查的前四条确实在扮演客服。说明新运行同样会出现此类问题。此时正式教师新增采样尚未开始。

为了避免自动进入教师采样，已暂停监督器父进程 228105；当前准入子进程 267110 继续完成在途盲解，未中断其请求或重置 deadline。暂停证据见新轮 execution/user-role-investigation-pause.json。supervisor.json 仍记录原来的 RUNNING 阶段，实际控制状态应同时查看此暂停文件和进程状态。

## 小规模功能对照

八次诊断请求均计入新轮累计预算（8250 tokens），不属于教师轨迹，不进入 SFT。两个 system-only canary 均准确返回指定内容，说明端点能够使用 system 指令；这不证明不存在任何网关内部改写。

三个历史异常请求重放：两个返回正常客户回复，一个为空；三个加入明确角色边界和 BANK REPRESENTATIVE 标签的请求均返回客户回复。样本太小，且原异常没有稳定复现，不能宣称修复有效或把 temperature=0 视为确定性保证。

## 修复建议

1. 保留 flip_roles 的 API 协议，给合成用户明确客户身份、客服对话方和服务职责边界；说明 reason-for-call 是客户要求银行完成的业务，而不是让模拟器自己执行客服职责。
2. 质量审核逐条检查 user 消息角色，提供用户消息编号与判据；要求身份验证、承诺银行侧办理、宣称查询银行内部记录等角色越界须拒绝。业务最后成功或 Agent 后来纠正不能覆盖前面的违规。
3. 对旧样本重新审核，保留原证据，确认异常和未决候选不进入严格主训练集。不要直接删除用户回合或拼接修复后的教师上下文；user loss=0 也不能消除对下一条受监督 assistant 回复的影响。
4. 3000 条目标应以复审合格的旧轨迹和新增合格独立轨迹累计。修改“旧 890 条全部必须保留”的交付假设，继续维持四次 seed、失败记录、配额和累计预算。

证据目录：`data/synthesis/targeted-native-v03-r10-repair8/verification/user-request-audit/`；当前端点请求筛查和对照结果在 expand3k 轮的 verification/ 下。此次核查没有修改生产用户提示词或原 SFT。
