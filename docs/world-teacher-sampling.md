# 独立验证与教师采样

`RolloutSettings.agent_models` 继续指定两个独立验证模型，用于任务改写审核、无答案解题和交叉检查。教师模型通过单独的 `TeacherPolicy` 选择，不修改已验收种子的模型设置。

教师策略文件示例：

```json
{"model": "openai/GLM-5.3-Flash", "seeds": [42, 43]}
```

`run_world_streaming.py --teacher-policy <文件>` 或 `stream_tasks(..., teacher_policy=TeacherPolicy(...))` 使用指定模型和两个不同 seed 采样。教师必须属于已验证模型集合。未指定策略时保留原来的双教师行为。用户模拟器与评分模型仍由 rollout 配置指定。

切换策略必须使用新的输出目录和显式迁移记录，不能覆盖已有 manifest。教师策略及实现代码进入生产身份；历史 task、答案、评分、工具输出和训练/验证划分保持原有证据绑定，复用前会检查。

复用轨迹要求任务、教师模型、seed 和有效模型设置一致，并严格回放工具结果。合格的旧 GLM 轨迹可以进入新 SFT；旧 Gemini 轨迹仍保存在历史输出中。真实失败不会因为切换模型而重新评分，同一 GLM trial 的中断也不会被静默重采。跨代拒绝记录持续有效。

新 SFT 在 `<output>/sft/shards/`，只包含训练任务的合格指定教师轨迹。`teacher-trials/<slot>/report.json` 记录各 seed 未通过或未完成的原因。部分教师失败时 task 的既有验收记录保留，但不会虚增 SFT 数量；task 和 SFT 目标均达成才标记 COMPLETE。

后台迁移入口 `run_world_teacher_stream.py --deployment <文件>` 校验固定代码、种子运行时、原设置和源 manifest，并记录实际同步/异步 HTTP 容量。所有参与请求共享 `TAU3_LLM_POOL`，以 `TAU3_LLM_CONCURRENCY` 限制 agent、user 和 judge 总请求并发。历史审计调用及物理 rollout 预算不会清零。
