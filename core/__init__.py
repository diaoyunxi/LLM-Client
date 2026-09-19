"""
LLM 客户端核心模块
包含后端连接、对话管理、工具系统和智能体循环
"""

from .backend import Backend, OllamaBackend, LlamaCppBackend
from .conversation import Conversation, Message
from .agent import AgentLoop
from .tools.loader import ToolLoader

__all__ = [
    "Backend",
    "OllamaBackend",
    "LlamaCppBackend",
    "Conversation",
    "Message",
    "AgentLoop",
    "ToolLoader",
]
