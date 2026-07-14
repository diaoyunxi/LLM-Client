"""
图形用户界面 (GUI)
使用 PyQt6，提供完整的桌面应用体验
"""

import os
import sys
import json
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backend import OllamaBackend, LlamaCppBackend, ModelInfo
from core.conversation import Conversation, Message
from core.agent import AgentLoop
from core.tools.loader import ToolLoader

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel, QSplitter,
        QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QTabWidget,
        QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
        QDialog, QDialogButtonBox, QProgressBar, QSystemTrayIcon, QMenu,
        QInputDialog, QPlainTextEdit, QFrame, QScrollArea, QSizePolicy
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
    from PyQt6.QtGui import QAction, QFont, QIcon, QPixmap, QImage
except ImportError:
    print("请先安装 PyQt6: pip install PyQt6")
    sys.exit(1)


class ChatWorker(QThread):
    """后台对话线程"""
    chunk_ready = pyqtSignal(str)
    tool_called = pyqtSignal(str, str)  # name, args
    tool_result = pyqtSignal(str, str)  # name, result
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, agent: AgentLoop, text: str, images: List[str] = None):
        super().__init__()
        self.agent = agent
        self.text = text
        self.images = images or []
        self._is_running = True

    def run(self):
        try:
            for chunk in self.agent.run(self.text, images=self.images, stream=True):
                if not self._is_running:
                    break
                self.chunk_ready.emit(chunk)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False
        self.wait(1000)


class MessageBubble(QFrame):
    """消息气泡组件"""

    def __init__(self, message: Message, parent=None):
        super().__init__(parent)
        self.message = message
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)

        # 角色标签
        role_text = {
            "user": "用户",
            "assistant": "AI",
            "system": "系统",
            "tool": "工具",
        }.get(self.message.role, self.message.role)

        self.role_label = QLabel(f"<b>{role_text}</b>")
        self.role_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.role_label)

        # 图片显示
        if self.message.images:
            img_label = QLabel(f"[图片附件: {len(self.message.images)} 张]")
            img_label.setStyleSheet("color: #2196F3; font-style: italic; font-size: 12px;")
            layout.addWidget(img_label)

        # 内容
        self.content_edit = QPlainTextEdit()
        self.content_edit.setPlainText(self.message.content)
        self.content_edit.setReadOnly(True)
        self.content_edit.setMaximumBlockCount(0)

        # 根据角色设置样式
        if self.message.role == "user":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #E3F2FD;
                    border-radius: 8px;
                    border: 1px solid #BBDEFB;
                    margin: 4px 40px 4px 4px;
                }
            """)
            self.content_edit.setStyleSheet("background: transparent; border: none;")
        elif self.message.role == "assistant":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #F5F5F5;
                    border-radius: 8px;
                    border: 1px solid #E0E0E0;
                    margin: 4px 4px 4px 40px;
                }
            """)
            self.content_edit.setStyleSheet("background: transparent; border: none;")
        elif self.message.role == "system":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #FFF3E0;
                    border-radius: 8px;
                    border: 1px solid #FFE0B2;
                    margin: 4px;
                }
            """)
            self.content_edit.setStyleSheet("background: transparent; border: none;")
        elif self.message.role == "tool":
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #E8F5E9;
                    border-radius: 8px;
                    border: 1px solid #C8E6C9;
                    margin: 4px 20px;
                }
            """)
            self.content_edit.setStyleSheet("background: transparent; border: none;")

        self.content_edit.setSizeAdjustPolicy(QPlainTextEdit.SizeAdjustPolicy.AdjustToContents)
        layout.addWidget(self.content_edit)

    def append_text(self, text: str):
        """追加文本（流式输出用）"""
        self.message.content += text
        self.content_edit.setPlainText(self.message.content)
        # 调整高度
        doc_height = self.content_edit.document().size().height()
        self.content_edit.setMinimumHeight(int(doc_height) + 20)


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["ollama", "llamacpp"])
        form.addRow("后端类型:", self.backend_combo)

        self.host_input = QLineEdit("localhost")
        form.addRow("主机地址:", self.host_input)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(11434)
        form.addRow("端口:", self.port_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        form.addRow("温度:", self.temp_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 50)
        self.max_iter_spin.setValue(10)
        form.addRow("最大迭代次数:", self.max_iter_spin)

        self.tools_dir_input = QLineEdit("tools")
        form.addRow("工具目录:", self.tools_dir_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM 客户端")
        self.setMinimumSize(1200, 800)

        self.backend = None
        self.conversation = Conversation(id="gui_session", title="GUI 对话")
        self.tool_loader = ToolLoader()
        self.agent = None
        self.current_model = ""
        self.image_attachments: List[str] = []
        self.chat_worker: Optional[ChatWorker] = None

        self._setup_ui()
        self._setup_menu()
        self._load_tools()

    def _setup_ui(self):
        """构建界面"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === 左侧边栏 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 连接设置
        conn_group = QGroupBox("连接设置")
        conn_layout = QFormLayout(conn_group)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Ollama", "llama.cpp"])
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        conn_layout.addRow("后端:", self.backend_combo)

        self.host_input = QLineEdit("localhost")
        conn_layout.addRow("主机:", self.host_input)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(11434)
        conn_layout.addRow("端口:", self.port_spin)

        self.connect_btn = QPushButton("🔄 连接")
        self.connect_btn.clicked.connect(self.do_connect)
        conn_layout.addRow(self.connect_btn)

        left_layout.addWidget(conn_group)

        # 模型选择
        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)

        self.model_info_label = QLabel("未连接")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #666; font-size: 11px;")
        model_layout.addWidget(self.model_info_label)

        left_layout.addWidget(model_group)

        # 工具列表
        tools_group = QGroupBox("已加载工具")
        tools_layout = QVBoxLayout(tools_group)

        self.tools_list = QListWidget()
        tools_layout.addWidget(self.tools_list)

        left_layout.addWidget(tools_group)

        # 系统提示词
        sys_group = QGroupBox("系统提示词")
        sys_layout = QVBoxLayout(sys_group)

        self.system_input = QLineEdit()
        self.system_input.setPlaceholderText("输入系统提示词...")
        sys_layout.addWidget(self.system_input)

        self.apply_sys_btn = QPushButton("应用")
        self.apply_sys_btn.clicked.connect(self.apply_system_prompt)
        sys_layout.addWidget(self.apply_sys_btn)

        left_layout.addWidget(sys_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 1)

        # === 中间聊天区域 ===
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(8)

        # 聊天记录滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_layout.setSpacing(8)
        self.messages_layout.addStretch()

        scroll.setWidget(self.messages_container)
        chat_layout.addWidget(scroll)

        # 图片附件显示
        self.image_label = QLabel("")
        self.image_label.setStyleSheet("color: #2196F3; font-size: 12px;")
        chat_layout.addWidget(self.image_label)

        # 输入区域
        input_layout = QHBoxLayout()

        self.attach_btn = QPushButton("📎")
        self.attach_btn.setToolTip("附加图片")
        self.attach_btn.setMaximumWidth(40)
        self.attach_btn.clicked.connect(self.attach_image)
        input_layout.addWidget(self.attach_btn)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入消息并按 Enter 发送...")
        self.input_box.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_box)

        self.send_btn = QPushButton("发送")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("⏹️")
        self.stop_btn.setToolTip("停止生成")
        self.stop_btn.setMaximumWidth(40)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        input_layout.addWidget(self.stop_btn)

        chat_layout.addLayout(input_layout)

        splitter.addWidget(chat_panel)
        splitter.setStretchFactor(1, 4)

        # === 右侧历史/信息 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        history_group = QGroupBox("对话历史")
        history_layout = QVBoxLayout(history_group)

        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)

        new_chat_btn = QPushButton("🆕 新对话")
        new_chat_btn.clicked.connect(self.new_conversation)
        history_layout.addWidget(new_chat_btn)

        right_layout.addWidget(history_group)

        # 参数调整
        params_group = QGroupBox("生成参数")
        params_layout = QFormLayout(params_group)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        params_layout.addRow("温度:", self.temp_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 50)
        self.max_iter_spin.setValue(10)
        params_layout.addRow("最大迭代:", self.max_iter_spin)

        right_layout.addWidget(params_group)

        status_label = QLabel("状态: 就绪")
        status_label.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(status_label)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(2, 1)

    def _setup_menu(self):
        """菜单栏"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")

        new_action = QAction("新对话", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_conversation)
        file_menu.addAction(new_action)

        save_action = QAction("保存对话", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_conversation)
        file_menu.addAction(save_action)

        load_action = QAction("加载对话", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_conversation)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("编辑")

        clear_action = QAction("清空对话", self)
        clear_action.triggered.connect(self.clear_chat)
        edit_menu.addAction(clear_action)

        tools_menu = menubar.addMenu("工具")

        reload_tools_action = QAction("重新加载工具", self)
        reload_tools_action.triggered.connect(self.reload_tools)
        tools_menu.addAction(reload_tools_action)

    def _load_tools(self):
        """加载工具"""
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
        if os.path.isdir(tools_dir):
            self.tool_loader.add_tools_dir(tools_dir)
            self.tool_loader.load_all()
        self.update_tools_list()

    def update_tools_list(self):
        """更新工具列表显示"""
        self.tools_list.clear()
        for name, tool in self.tool_loader.tools.items():
            item = QListWidgetItem(f"{name}: {tool.description}")
            self.tools_list.addItem(item)

    def on_backend_changed(self, text: str):
        """后端类型改变"""
        if text == "llama.cpp":
            self.port_spin.setValue(8080)
        else:
            self.port_spin.setValue(11434)

    def do_connect(self):
        """连接后端"""
        backend_type = self.backend_combo.currentText().lower()
        host = self.host_input.text().strip()
        port = self.port_spin.value()

        if backend_type == "ollama":
            self.backend = OllamaBackend(host=host, port=port)
        else:
            self.backend = LlamaCppBackend(host=host, port=port)

        models = self.backend.list_models()
        self.model_combo.clear()

        if not models:
            QMessageBox.warning(self, "警告", "未检测到模型，请确认后端服务已启动。")
            return

        for m in models:
            self.model_combo.addItem(m.name)

        self.model_combo.setEnabled(True)
        self.current_model = models[0].name
        self.model_combo.setCurrentText(self.current_model)

        info = models[0]
        vision = "支持" if info.supports_vision else "不支持"
        tools = "支持" if info.supports_tools else "不支持"
        self.model_info_label.setText(f"视觉: {vision} | 工具: {tools}")

        QMessageBox.information(self, "连接成功", f"已连接到 {backend_type}，发现 {len(models)} 个模型。")

    def apply_system_prompt(self):
        """应用系统提示词"""
        text = self.system_input.text().strip()
        if text:
            self.conversation.add_system_message(text)
            self.add_message_to_ui(Message(role="system", content=text))

    def add_message_to_ui(self, message: Message):
        """添加消息到界面"""
        bubble = MessageBubble(message)
        # 插入到 stretch 之前
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        return bubble

    def send_message(self):
        """发送消息"""
        if not self.backend or not self.current_model:
            QMessageBox.warning(self, "警告", "请先连接后端并选择模型。")
            return

        text = self.input_box.text().strip()
        if not text:
            return

        self.input_box.clear()

        # 初始化 Agent
        if not self.agent:
            self.agent = AgentLoop(
                backend=self.backend,
                conversation=self.conversation,
                tool_loader=self.tool_loader,
                model=self.current_model,
                max_iterations=self.max_iter_spin.value(),
                temperature=self.temp_spin.value(),
            )
        else:
            self.agent.max_iterations = self.max_iter_spin.value()
            self.agent.temperature = self.temp_spin.value()

        # 添加用户消息
        self.conversation.add_message("user", text, images=self.image_attachments)
        self.add_message_to_ui(self.conversation.messages[-1])

        # 更新历史
        self.update_history_list()

        # 禁用输入
        self.input_box.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 启动工作线程
        self.current_ai_bubble = MessageBubble(Message(role="assistant", content=""))
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self.current_ai_bubble)

        self.chat_worker = ChatWorker(self.agent, text, self.image_attachments)
        self.chat_worker.chunk_ready.connect(self.on_chunk_ready)
        self.chat_worker.finished_signal.connect(self.on_chat_finished)
        self.chat_worker.error_signal.connect(self.on_chat_error)
        self.chat_worker.start()

        # 清空图片
        self.image_attachments = []
        self.image_label.setText("")

    def on_chunk_ready(self, chunk: str):
        """收到新内容"""
        self.current_ai_bubble.append_text(chunk)
        # 滚动到底部
        self.messages_container.parent().parent().verticalScrollBar().setValue(
            self.messages_container.parent().parent().verticalScrollBar().maximum()
        )

    def on_chat_finished(self):
        """对话完成"""
        self.input_box.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_box.setFocus()

    def on_chat_error(self, error: str):
        """对话错误"""
        self.current_ai_bubble.append_text(f"\n[错误] {error}")
        self.on_chat_finished()

    def stop_generation(self):
        """停止生成"""
        if self.chat_worker:
            self.chat_worker.stop()
            self.chat_worker = None
        self.on_chat_finished()

    def attach_image(self):
        """附加图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.gif *.bmp *.webp)"
        )
        if files:
            for f in files:
                try:
                    encoded = self.backend.encode_image(f)
                    self.image_attachments.append(encoded)
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"加载图片失败: {e}")
            self.image_label.setText(f"📎 已附加 {len(self.image_attachments)} 张图片")

    def new_conversation(self):
        """新对话"""
        self.conversation = Conversation(id="gui_session", title="GUI 对话")
        self.agent = None
        # 清空消息
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.history_list.clear()

    def clear_chat(self):
        """清空当前对话显示"""
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.conversation.clear()

    def save_conversation(self):
        """保存对话"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存对话", f"conversation_{self.conversation.id}.json", "JSON (*.json)"
        )
        if path:
            try:
                self.conversation.save(path)
                QMessageBox.information(self, "成功", "对话已保存。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def load_conversation(self):
        """加载对话"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载对话", "", "JSON (*.json)"
        )
        if path:
            try:
                self.conversation = Conversation.load(path)
                self.agent = None
                # 刷新显示
                while self.messages_layout.count() > 1:
                    item = self.messages_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                for msg in self.conversation.messages:
                    self.add_message_to_ui(msg)
                self.update_history_list()
                QMessageBox.information(self, "成功", "对话已加载。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def update_history_list(self):
        """更新历史列表"""
        self.history_list.clear()
        for i, msg in enumerate(self.conversation.messages):
            if msg.role in ("user", "assistant"):
                preview = msg.content[:30] + "..." if len(msg.content) > 30 else msg.content
                role = "用户" if msg.role == "user" else "AI"
                self.history_list.addItem(f"{role}: {preview}")

    def reload_tools(self):
        """重新加载工具"""
        self.tool_loader = ToolLoader()
        self._load_tools()
        QMessageBox.information(self, "完成", f"已加载 {len(self.tool_loader.tools)} 个工具。")


def run_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
