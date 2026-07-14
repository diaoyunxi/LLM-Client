"""
多轮对话管理模块
管理消息历史、上下文窗口、会话持久化

后续可扩展: 对话摘要功能 —— 当消息数超过阈值时, 自动调用 LLM 生成
历史对话摘要, 替换原始消息以节省 token 预算。当前版本仅做窗口截断。
"""

import json
import re
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

    def estimate_tokens(self) -> int:
        """
        估算当前对话的 token 数量 (简单估算)

        估算规则:
        - 中文按字符数计算 (每个中文字符约 1-2 token)
        - 英文按空格分词数计算 (每个单词约 1-2 token)
        - 混合文本中分别统计中文字符和英文单词

        Returns:
            估算的 token 数
        """
        total = 0
        for msg in self.messages:
            text = msg.content or ""
            if not text:
                continue
            # 统计中文字符数
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            # 去除中文后按空格分词统计英文单词数
            non_chinese = re.sub(r'[\u4e00-\u9fff]', ' ', text)
            english_words = len(non_chinese.split())
            # 中文约 1.5 token/字, 英文约 1.5 token/词
            total += int(chinese_chars * 1.5 + english_words * 1.5)
        # 加上 system prompt 的 token
        if self.system_prompt:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.system_prompt))
            non_chinese = re.sub(r'[\u4e00-\u9fff]', ' ', self.system_prompt)
            english_words = len(non_chinese.split())
            total += int(chinese_chars * 1.5 + english_words * 1.5)
        return total

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
