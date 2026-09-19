# LLM 客户端

一个支持 **Ollama** 和 **llama.cpp** 的通用 LLM 客户端，具备多模态支持、外置自定义工具、智能体循环和多轮对话能力。

提供三种界面模式：**CLI（纯终端）**、**TUI（终端界面）**、**GUI（图形界面）**。

---

## 功能特性

- **双后端支持**：同时兼容 Ollama（本地/远程）和 llama.cpp HTTP Server
- **多模态**：支持图片上传，供视觉模型（如 LLaVA）分析
- **外置工具系统**：工具定义外置在独立 Python 文件中，通过文件头部注释声明，热加载无需重启
- **智能体循环**：自动检测模型输出的工具调用，执行后自动将结果带回对话，支持多轮工具调用
- **三种界面**：
  - CLI：最轻量，无额外依赖
  - TUI（Textual）：终端内的丰富界面
  - GUI（PyQt6）：完整桌面应用体验
- **对话管理**：多轮上下文、持久化保存/加载、系统提示词设置

---

## 项目结构

```
llm_client/
├── core/                   # 后端核心模块
│   ├── backend.py          # 后端抽象（OllamaBackend / LlamaCppBackend）
│   ├── conversation.py     # 多轮对话管理
│   ├── agent.py            # 智能体循环（工具调用解析与执行）
│   └── tools/              # 工具系统
│       ├── parser.py       # 从注释解析工具定义
│       └── loader.py       # 工具加载与执行
├── interfaces/             # 三种界面实现
│   ├── cli.py              # 纯终端命令行
│   ├── tui.py              # Textual 终端界面
│   └── gui.py              # PyQt6 图形界面
├── tools/                  # 外置工具目录（示例工具存放处）
│   ├── calculator.py       # 计算器工具
│   ├── weather.py          # 天气查询工具（模拟）
│   └── datetime_tool.py    # 日期时间工具
├── main.py                 # 统一入口
├── requirements.txt        # 依赖
└── README.md               # 本文件
```

---

## 安装

### 1. 克隆或下载项目

```bash
cd llm_client
```

### 2. 安装依赖

根据你想使用的界面，安装对应依赖：

```bash
# 基础依赖（CLI 模式必需）
pip install ollama requests

# TUI 模式需要
pip install textual

# GUI 模式需要
pip install PyQt6

# 安装全部依赖
pip install -r requirements.txt
```

### 3. 准备后端

**Ollama**：
```bash
# 安装并启动 Ollama
ollama serve
# 拉取模型
ollama pull llama3.1
ollama pull llava  # 视觉模型
```

**llama.cpp**：
```bash
# 编译并启动 server
./server -m model.gguf --port 8080
```

---

## 使用方法

### 统一入口

```bash
# CLI 模式（默认）
python main.py --mode cli --backend ollama --model llama3.1

# 启用思考过程显示（需模型支持，如 DeepSeek-R1、Qwen3）
python main.py --mode cli --backend ollama --model deepseek-r1 --think

# TUI 模式（默认启用思考过程显示）
python main.py --mode tui --backend ollama

# GUI 模式（右侧参数面板可切换"显示思考过程"复选框）
python main.py --mode gui --backend ollama

# 连接远程 llama.cpp
python main.py --mode gui --backend llamacpp --host 192.168.1.100 --port 8080

# 带图片的多模态对话（CLI）
python main.py --mode cli --backend ollama --model llava --image ./photo.png

# 指定系统提示词
python main.py --mode cli --system "你是一个专业的编程助手"
```

### CLI 命令

在 CLI 模式下，输入 `/help` 查看可用命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/quit` | 退出程序 |
| `/clear` | 清空对话历史 |
| `/save [路径]` | 保存对话到 JSON 文件 |
| `/load [路径]` | 从 JSON 文件加载对话 |
| `/image <路径>` | 加载图片（多模态） |
| `/models` | 列出可用模型 |
| `/tools` | 列出已加载的工具 |

---

## 自定义工具开发

工具文件是外置的 Python 脚本，放置在 `tools/` 目录下即可自动加载。

### 工具文件格式

```python
"""
TOOL_NAME: 工具名（唯一标识）
TOOL_DESCRIPTION: 工具功能描述（模型通过此描述决定何时调用）
TOOL_PARAMETERS:
    参数名:
        type: 参数类型（string/integer/number/boolean/array/object）
        description: 参数说明
        required: true/false
        default: 默认值（可选）
        enum: [可选值列表]（可选）
"""


def run(参数名: 类型, ...):
    """
    工具执行函数，函数名必须是 run
    返回值会被序列化为 JSON 传给模型
    """
    # 你的逻辑
    return {"result": "..."}
```

### 示例：创建一个查询 IP 的工具

创建 `tools/ip_query.py`：

```python
"""
TOOL_NAME: ip_query
TOOL_DESCRIPTION: 查询 IP 地址的地理位置信息
TOOL_PARAMETERS:
    ip:
        type: string
        description: 要查询的 IP 地址，如 "8.8.8.8"
        required: true
"""

import requests


def run(ip: str):
    try:
        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=10)
        data = resp.json()
        return {
            "ip": ip,
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country_name"),
            "org": data.get("org"),
        }
    except Exception as e:
        return {"error": str(e)}
```

保存后重启客户端或执行重新加载工具即可使用。

---

## 工具调用流程

1. **加载阶段**：启动时扫描 `tools/` 目录，解析每个 Python 文件头部的 `TOOL_` 注释
2. **对话阶段**：将工具定义随消息一起发送给模型（Ollama 原生支持；llama.cpp 通过 prompt 注入）
3. **检测阶段**：模型输出中若包含工具调用（JSON/XML/代码块格式），自动解析
4. **执行阶段**：调用对应的 Python 函数，验证参数，执行并捕获结果
5. **循环阶段**：将工具结果以 `tool` 角色加入对话历史，再次请求模型，直到不再调用工具

---

## 配置说明

### 思考过程显示（Thinking）

支持具备推理能力的模型（如 DeepSeek-R1、Qwen3、DeepSeek-v3.1、GPT-OSS），可实时流式显示模型的思考过程：

- **CLI 模式**：添加 `--think` 参数启用，思考内容以灰色斜体显示
- **TUI 模式**：默认启用，思考过程以独立区域显示
- **GUI 模式**：右侧参数面板的"显示思考过程"复选框控制，默认勾选；思考过程以可折叠区域显示（点击展开/折叠）

### Ollama 后端

- 默认地址：`http://localhost:11434`
- 支持工具调用（需模型支持，如 llama3.1、qwen2.5 等）
- 支持多模态（需视觉模型，如 llava、moondream 等）

### llama.cpp 后端

- 默认地址：`http://localhost:8080`
- 工具调用通过 prompt 注入实现（兼容性取决于模型能力）
- 多模态支持取决于编译选项和模型

---

## 注意事项

1. **模型能力**：工具调用效果取决于模型本身的能力，建议使用经过工具调用微调的模型（如 llama3.1、qwen2.5）
2. **安全**：外置工具直接执行 Python 代码，请确保 `tools/` 目录下的文件可信
3. **上下文长度**：多轮对话和工具结果会占用大量上下文，长对话建议适当清空历史
4. **图片大小**：多模态图片会转为 base64，大图片可能导致请求变慢

---

## 依赖版本

| 包 | 最低版本 | 说明 |
|----|---------|------|
| Python | 3.8+ | 必需 |
| ollama | 0.3.0+ | Ollama 后端 |
| requests | 2.31.0+ | HTTP 请求 |
| PyQt6 | 6.6.0+ | GUI 模式 |
| textual | 0.52.0+ | TUI 模式 |
| pillow | 10.0.0+ | 图片处理 |

---

## License

MIT License
