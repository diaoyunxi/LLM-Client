# Contributing to LLM-Client

感谢你对 LLM-Client 项目的关注！

## 如何贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 代码规范

- Python 代码遵循 PEP 8，使用 ruff 进行格式化
- 提交信息使用中文或英文均可
- PR 描述中说明改动内容和原因

## 项目架构

\`\`\`
LLM-Client/
├── main.py          # 统一入口
├── core/            # 核心模块（后端抽象层、智能体循环）
├── interfaces/      # 界面层（CLI/TUI/GUI）
├── tools/           # 外置工具
└── config/          # 配置文件
\`\`\`
