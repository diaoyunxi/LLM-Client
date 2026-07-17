"""
终端用户界面 (TUI)
使用 Textual 框架，提供比 CLI 更丰富的交互体验
"""

import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backend import OllamaBackend, LlamaCppBackend, ModelInfo, StreamChunk
from core.conversation import Conversation, Message
from core.agent import AgentLoop
from core.tools.loader import ToolLoader

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Header, Footer, Input, Static, Button, Select, Label,
        ListView, ListItem, TextArea, Markdown, LoadingIndicator,
        TabbedContent, TabPane
    )
    from textual.containers import Horizontal, Vertical, VerticalScroll, Container
    from textual.reactive import reactive
    from textual.worker import Worker
except ImportError:
    print("请先安装 textual: pip install textual")
    sys.exit(1)


class ChatMessageWidget(Static):
    """单条消息显示组件，支持思考过程折叠显示"""

    def __init__(self, message: Message, **kwargs):
        self.msg = message
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        role_label = {
            "user": "🧑 用户",
            "assistant": "🤖 AI",
            "system": "⚙️ 系统",
            "tool": "🔧 工具",
        }.get(self.msg.role, self.msg.role)

        with Container(classes=f"msg-container msg-{self.msg.role}"):
            yield Label(f"[{role_label}]", classes="msg-role")
            if self.msg.images:
                yield Label(f"[图片: {len(self.msg.images)} 张]", classes="msg-image")
            # 思考过程区域（灰色折叠显示）
            if self.msg.thinking:
                with Container(classes="thinking-container"):
                    yield Label("💭 思考过程", classes="thinking-label")
                    yield Static(self.msg.thinking, classes="thinking-content")
            yield Markdown(self.msg.content, classes="msg-content")


class LLMClientTUI(App):
    """TUI 主应用"""

    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
        layout: grid;
        grid-size: 3;
        grid-columns: 25fr 50fr 25fr;
    }

    #sidebar-left {
        width: 100%;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    #chat-area {
        width: 100%;
        height: 100%;
        border: solid $primary;
        layout: vertical;
    }

    #messages-scroll {
        width: 100%;
        height: 1fr;
        padding: 1;
    }

    #input-area {
        width: 100%;
        height: auto;
        dock: bottom;
        padding: 1;
    }

    #user-input {
        width: 1fr;
    }

    #send-btn {
        width: auto;
    }

    #sidebar-right {
        width: 100%;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    .msg-container {
        padding: 1;
        margin: 1 0;
        border: solid $surface-lighten-1;
    }

    .msg-user {
        background: $surface-darken-1;
    }

    .msg-assistant {
        background: $surface;
    }

    .msg-system {
        background: $warning-darken-2;
        color: $text;
    }

    .msg-tool {
        background: $success-darken-2;
        color: $text;
    }

    .msg-role {
        text-style: bold;
        margin-bottom: 1;
    }

    .msg-content {
        padding: 1;
    }

    .msg-image {
        color: $accent;
        text-style: italic;
    }

    .thinking-container {
        margin: 0 1;
        padding: 1;
        border: dashed $surface-lighten-1;
        background: $surface-darken-2;
    }

    .thinking-label {
        color: $warning-darken-1;
        text-style: bold italic;
        margin-bottom: 1;
    }

    .thinking-content {
        color: $text-disabled;
        dock: top;
        height: auto;
    }

    #settings-container {
        padding: 2;
    }

    .setting-row {
        margin: 1 0;
        height: auto;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        ("ctrl+s", "settings", "设置"),
        ("ctrl+c", "clear_chat", "清空"),
    ]

    backend_type = reactive("ollama")
    host = reactive("localhost")
    port = reactive(11434)
    current_model = reactive("")
    models: list[ModelInfo] = []

    def __init__(self):
        super().__init__()
        self.backend = None
        self.conversation = Conversation(id="tui_session", title="TUI 对话")
        self.tool_loader = ToolLoader()
        self.agent = None
        self.image_attachments: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # 左侧边栏 - 模型和设置
            with Vertical(id="sidebar-left"):
                yield Label("⚙️ 设置", classes="h2")
                yield Static("后端类型:")
                yield Select(
                    [("Ollama", "ollama"), ("llama.cpp", "llamacpp")],
                    value="ollama",
                    id="backend-select",
                )
                yield Static("主机:")
                yield Input(value="localhost", id="host-input")
                yield Static("端口:")
                yield Input(value="11434", id="port-input")
                yield Button("🔄 连接", id="connect-btn")
                yield Static("")
                yield Static("📋 模型:")
                yield Select([], id="model-select", prompt="先连接后端")
                yield Static("")
                yield Button("📁 加载图片", id="image-btn")
                yield Static(id="image-status")

            # 中间 - 聊天区域
            with Vertical(id="chat-area"):
                with VerticalScroll(id="messages-scroll"):
                    yield Static("💬 欢迎使用 LLM 客户端\n请先连接后端并选择模型。", id="welcome-msg")
                with Horizontal(id="input-area"):
                    yield Input(placeholder="输入消息... (Ctrl+Enter 发送)", id="user-input")
                    yield Button("发送", id="send-btn", variant="primary")

            # 右侧边栏 - 工具和状态
            with Vertical(id="sidebar-right"):
                yield Label("🔧 工具", classes="h2")
                yield Static(id="tools-list", content="未加载工具")
                yield Static("")
                yield Label("📊 状态", classes="h2")
                yield Static(id="status-text", content="未连接")

        yield Footer()

    def on_mount(self) -> None:
        """应用加载时"""
        self.title = "LLM 客户端"
        self.sub_title = "TUI 界面"
        # 加载工具
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
        if os.path.isdir(tools_dir):
            self.tool_loader.add_tools_dir(tools_dir)
            loaded = self.tool_loader.load_all()
            self.update_tools_display()

    def update_tools_display(self) -> None:
        """更新工具列表显示"""
        tools_text = self.query_one("#tools-list", Static)
        if self.tool_loader.tools:
            lines = [f"  • {name}: {t.description}" for name, t in self.tool_loader.tools.items()]
            tools_text.update("\n".join(lines))
        else:
            tools_text.update("未加载工具")

    def update_status(self, text: str) -> None:
        """更新状态显示"""
        self.query_one("#status-text", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击事件"""
        btn_id = event.button.id

        if btn_id == "connect-btn":
            self.do_connect()
        elif btn_id == "send-btn":
            self.do_send()
        elif btn_id == "image-btn":
            self.do_attach_image()

    def do_connect(self) -> None:
        """连接后端"""
        backend_type = self.query_one("#backend-select", Select).value
        host = self.query_one("#host-input", Input).value.strip()
        port_str = self.query_one("#port-input", Input).value.strip()

        try:
            port = int(port_str)
        except ValueError:
            self.update_status("端口格式错误")
            return

        self.update_status("连接中...")

        if backend_type == "ollama":
            self.backend = OllamaBackend(host=host, port=port)
        else:
            self.backend = LlamaCppBackend(host=host, port=port)

        self.models = self.backend.list_models()
        if not self.models:
            self.update_status("未检测到模型")
            return

        # 更新模型选择器
        model_select = self.query_one("#model-select", Select)
        options = [(m.name, m.name) for m in self.models]
        model_select.set_options(options)
        self.current_model = self.models[0].name
        model_select.value = self.current_model

        self.update_status(f"已连接 | {len(self.models)} 个模型")
        self.notify("连接成功！", severity="information")

    def on_select_changed(self, event: Select.Changed) -> None:
        """选择器变化事件"""
        if event.select.id == "model-select":
            self.current_model = event.value or ""
        elif event.select.id == "backend-select":
            # 自动调整默认端口
            port_input = self.query_one("#port-input", Input)
            if event.value == "ollama":
                port_input.value = "11434"
            else:
                port_input.value = "8080"

    def do_attach_image(self) -> None:
        """附加图片（简化版，直接输入路径）"""
        # TUI 中简单使用输入框
        self.query_one("#user-input", Input).focus()
        self.notify("请在输入框中输入 /image <路径> 来加载图片", severity="warning")

    def do_send(self) -> None:
        """发送消息"""
        if not self.backend or not self.current_model:
            self.notify("请先连接后端并选择模型", severity="error")
            return

        input_widget = self.query_one("#user-input", Input)
        text = input_widget.value.strip()
        if not text:
            return

        input_widget.value = ""

        # 处理特殊命令
        if text.startswith("/"):
            self.handle_command(text)
            return

        # 初始化 Agent
        if not self.agent:
            self.conversation.model = self.current_model
            self.agent = AgentLoop(
                backend=self.backend,
                conversation=self.conversation,
                tool_loader=self.tool_loader,
                model=self.current_model,
                think=True,
            )

        # 添加用户消息到显示
        self.conversation.add_message("user", text, images=self.image_attachments)
        self.add_message_widget(self.conversation.messages[-1])

        # 清除欢迎消息
        welcome = self.query_one("#welcome-msg", Static)
        if welcome:
            welcome.remove()

        # 异步运行对话
        self.run_worker(self._chat_worker(text), group="chat", exclusive=True)

    async def _chat_worker(self, text: str) -> None:
        """聊天工作线程"""
        messages_scroll = self.query_one("#messages-scroll", VerticalScroll)

        # 创建 AI 消息占位
        ai_msg = Message(role="assistant", content="", thinking="")
        self.conversation.messages.append(ai_msg)
        msg_widget = ChatMessageWidget(ai_msg)
        await messages_scroll.mount(msg_widget)

        full_content = ""
        full_thinking = ""
        try:
            for chunk in self.agent.run(text, images=self.image_attachments, stream=True):
                if chunk.thinking:
                    full_thinking += chunk.thinking
                if chunk.content:
                    full_content += chunk.content
                ai_msg.content = full_content
                ai_msg.thinking = full_thinking
                # 刷新显示
                msg_widget.refresh()
        except Exception as e:
            full_content += f"\n[错误] {e}"
            ai_msg.content = full_content
            msg_widget.refresh()

        # 清空图片附件
        self.image_attachments = []
        self.query_one("#image-status", Static).update("")

    def add_message_widget(self, message: Message) -> None:
        """添加消息到聊天区域"""
        messages_scroll = self.query_one("#messages-scroll", VerticalScroll)
        widget = ChatMessageWidget(message)
        messages_scroll.mount(widget)
        messages_scroll.scroll_end()

    def handle_command(self, text: str) -> None:
        """处理命令"""
        parts = text.split(maxsplit=1)
        cmd = parts[0][1:].lower()

        if cmd == "clear":
            self.conversation.clear()
            messages_scroll = self.query_one("#messages-scroll", VerticalScroll)
            messages_scroll.remove_children()
            self.notify("对话已清空", severity="information")
        elif cmd == "image" and len(parts) > 1:
            path = parts[1].strip()
            if os.path.exists(path):
                try:
                    encoded = self.backend.encode_image(path)
                    self.image_attachments.append(encoded)
                    self.query_one("#image-status", Static).update(f"已加载: {os.path.basename(path)}")
                    self.notify(f"图片已加载: {path}", severity="information")
                except Exception as e:
                    self.notify(f"图片加载失败: {e}", severity="error")
            else:
                self.notify("文件不存在", severity="error")
        elif cmd == "save" and len(parts) > 1:
            try:
                self.conversation.save(parts[1].strip())
                self.notify("对话已保存", severity="information")
            except Exception as e:
                self.notify(f"保存失败: {e}", severity="error")
        elif cmd == "load" and len(parts) > 1:
            try:
                self.conversation = Conversation.load(parts[1].strip())
                if self.agent:
                    self.agent.conversation = self.conversation
                # 刷新显示
                messages_scroll = self.query_one("#messages-scroll", VerticalScroll)
                messages_scroll.remove_children()
                for msg in self.conversation.messages:
                    self.add_message_widget(msg)
                self.notify("对话已加载", severity="information")
            except Exception as e:
                self.notify(f"加载失败: {e}", severity="error")
        else:
            self.notify(f"未知命令: {cmd}", severity="warning")

    def action_settings(self) -> None:
        """打开设置"""
        self.notify("设置面板（TODO）", severity="information")

    def action_clear_chat(self) -> None:
        """清空对话"""
        self.conversation.clear()
        messages_scroll = self.query_one("#messages-scroll", VerticalScroll)
        messages_scroll.remove_children()
        self.notify("对话已清空", severity="information")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """输入框提交事件"""
        if event.input.id == "user-input":
            self.do_send()


def run_tui():
    app = LLMClientTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
