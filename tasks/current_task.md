当前任务：将 DMMP2 中最新强 DF/RF guidance 流程迁移到 DMMPv3，并统一中文文档、脚本入口和结果目录约束。

当前状态：

- DMMPv3 工程骨架已创建。
- DMMP2 的核心实现已迁移到 `dmmp/`。
- `dmmp/` 顶层业务代码已按类别放入子目录。
- `docs/implementation_index.md` 与 `docs/runbook.md` 已记录当前实现和运行入口。
- 正式方案统一称为 DMMPv3，不再使用旧版本编号命名。
- 新结果默认写入 `D:\learning\TOR\defence\DMMPv3\results`。
- 历史 fixed DF/RF checkpoint 仅作为已验证固定评测器引用，不移动、不覆盖。
- 本次未启动正式训练。
