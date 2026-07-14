"""
多轮对话管理模块
管理消息历史、上下文窗口、会话持久化
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .backend import ChatMessage


@dataclass
class Message:
    """内部消息表示"""
    role: str
    content: str
    images: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(
            role=self.role,
            content=self.content,
            images=self.images,
            tool_calls=self.metadata.get("tool_calls", []),
            tool_call_id=self.metadata.get("tool_call_id"),
        )

    @classmethod
    def from_chat_message(cls, msg: ChatMessage) -> "Message":
        return cls(
            role=msg.role,
            content=msg.content,
            images=msg.images,
            metadata={
                "tool_calls": msg.tool_calls,
                "tool_call_id": msg.tool_call_id,
            },
        )


@dataclass
class Conversation:
    """对话会话"""
    id: str
    title: str = "新对话"
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = ""
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"conv_{int(time.time() * 1000)}"

    def add_message(self, role: str, content: str, images: List[str] = None, **kwargs) -> Message:
        """添加消息"""
        msg = Message(
            role=role,
            content=content,
            images=images or [],
            metadata=kwargs,
        )
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def add_system_message(self, content: str) -> None:
        """设置系统提示词"""
        self.system_prompt = content
        # 如果第一条是 system，替换它
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = content
        else:
            self.messages.insert(0, Message(role="system", content=content))
        self.updated_at = time.time()

    def get_context_messages(self, max_messages: int = 50) -> List[ChatMessage]:
        """
        获取用于发送给模型的消息列表
        包含 system prompt 和最近的消息
        """
        result = []
        # 优先加入 system prompt
        if self.system_prompt:
            result.append(ChatMessage(role="system", content=self.system_prompt))

        # 取最近的消息
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        for msg in recent:
            # 跳过已经被 system_prompt 覆盖的原始 system 消息
            if msg.role == "system" and self.system_prompt and msg == self.messages[0]:
                continue
            result.append(msg.to_chat_message())
        return result

    def clear(self) -> None:
        """清空对话历史"""
        self.messages.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "images": m.images,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conv = cls(
            id=data.get("id", ""),
            title=data.get("title", "新对话"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {}),
        )
        for m in data.get("messages", []):
            conv.messages.append(Message(
                role=m["role"],
                content=m["content"],
                images=m.get("images", []),
                timestamp=m.get("timestamp", time.time()),
                metadata=m.get("metadata", {}),
            ))
        return conv

    def save(self, filepath: str) -> None:
        """保存对话到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Conversation":
        """从文件加载对话"""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
