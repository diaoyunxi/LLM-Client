> [← 返回文档目录](../README.md)

# 命令行参数（`main.py`）

统一入口 `python main.py`，参数由 `argparse` 定义；`interfaces/cli.py` 提供等价的 CLI 参数。

| 参数 | 说明 |
|---|---|
| `--mode` | 界面模式：`cli` / `tui` / `gui` |
| `--backend` | 后端：`ollama` / `llamacpp` |
| `--model` | 模型名称 |
| `--host` / `--port` | 后端地址与端口 |
| `--system` | 系统提示词 |
| `--image` | 上传图片路径（多模态） |
| `--think` | 启用思考过程显示 |
| `--tools-dir` | 外置工具目录，默认 `tools` |
| `--temperature` | 生成温度，浮点，默认 `0.7` |
| `--max-iterations` | 智能体最大迭代次数，整数，默认 `10` |

CLI 交互命令：`/help`、`/quit`、`/clear`、`/save [路径]`、`/load [路径]`、
`/image <路径>`、`/models`、`/tools`。

> 以上参数在 `main.py` 与 `interfaces/cli.py` 中均有 `add_argument` 定义。
