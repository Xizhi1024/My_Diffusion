# AGENTS.md

## Repository Map

A full codemap is available at `CODEMAP.md` in the project root（仓库代码全景地图，中文，覆盖 src/、scripts/、tests/、configs/、wuzhe/ 全部源码文件的职责/接口/依赖/输入输出/处理过程）。

Before working on any task, read `CODEMAP.md` to understand:

- Project architecture and entry points（第 0 章总览：SLMF-BBDM 架构、端到端数据流、pixi 任务表、实验演进脉络）
- Directory responsibilities and design patterns（第 1-6 章：数据层 / 模型核心 / 损失项 / 子包 / 机制验证库）
- Data flow and integration points between modules（第 7-12 章：训练评估入口、H1-H6 验证脚本、V2 管线、评估分析、云协作）
- Nested project and tests as contracts（第 13-14 章）；配置契约见第 15 章

For deep work on a specific folder, jump to the corresponding chapter（每章开头有模块概述，结尾有 grep 查证的模块依赖小结）。
