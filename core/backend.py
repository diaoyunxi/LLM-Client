"""
后端抽象层
支持 Ollama (通过 ollama 库) 和 llama.cpp (直接 HTTP 连接)
"""

import base64
import json
import requests
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import ollama


@dataclass
class ChatMessage:
    """统一消息格式"""
    role: str  # system / user / assistant / tool
    content: str = ""
    images: List[str] = field(default_factory=list)  # base64 编码的图片
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    size: int = 0
    parameter_size: str = ""
    format: str = ""
    families: List[str] = field(default_factory=list)
    supports_vision: bool = False
    supports_tools: bool = False


class Backend(ABC):
    """后端抽象基类"""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """获取可用模型列表"""
        pass

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        发起对话
        返回生成器，逐字输出回复内容
        """
        pass

    @abstractmethod
    def chat_complete(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发起对话，返回完整响应（用于工具调用解析）
        """
        pass

    def encode_image(self, image_path: str) -> str:
        """将图片转为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class OllamaBackend(Backend):
    """Ollama 后端（使用 ollama Python 库）"""

    def __init__(self, host: str = "localhost", port: int = 11434):
        super().__init__(host, port)
        self.client = ollama.Client(host=f"http://{host}:{port}")

    def list_models(self) -> List[ModelInfo]:
        models = []
        try:
            response = self.client.list()
            for m in response.get("models", []):
                info = ModelInfo(name=m.get("model", m.get("name", "unknown")))
                details = m.get("details", {})
                info.parameter_size = details.get("parameter_size", "")
                info.format = details.get("format", "")
                info.families = details.get("families", [])
                info.size = m.get("size", 0)
                # 根据模型名或 families 判断是否支持视觉
                vision_keywords = ["llava", "vision", "multimodal", "bakllava", "moondream"]
                name_lower = info.name.lower()
                info.supports_vision = any(k in name_lower for k in vision_keywords)
                info.supports_tools = True  # Ollama 支持工具调用
                models.append(info)
        except Exception as e:
            print(f"[OllamaBackend] 获取模型列表失败: {e}")
        return models

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """将内部消息格式转为 ollama 格式"""
        result = []
        for msg in messages:
            item = {"role": msg.role, "content": msg.content}
            if msg.images:
                item["images"] = msg.images
            if msg.tool_calls:
                item["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            result.append(item)
        return result

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        ollama_messages = self._convert_messages(messages)
        options = {"temperature": temperature}
        options.update(kwargs)

        try:
            response = self.client.chat(
                model=model,
                messages=ollama_messages,
                tools=tools,
                stream=True,
                options=options,
            )
            for chunk in response:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield f"\n[错误] Ollama 对话失败: {e}\n"

    def chat_complete(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        ollama_messages = self._convert_messages(messages)
        options = {"temperature": temperature}
        options.update(kwargs)

        try:
            response = self.client.chat(
                model=model,
                messages=ollama_messages,
                tools=tools,
                stream=False,
                options=options,
            )
            return response
        except Exception as e:
            return {"error": str(e), "message": {"content": f"[错误] Ollama 请求失败: {e}"}}


class LlamaCppBackend(Backend):
    """
    llama.cpp 后端（直接 HTTP 连接）
    兼容 llama.cpp server 的 /completion 和 /chat/completion 接口
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        super().__init__(host, port)
        self.base_url = f"http://{host}:{port}"

    def list_models(self) -> List[ModelInfo]:
        """llama.cpp server 通常只运行一个模型"""
        models = []
        try:
            resp = requests.get(f"{self.base_url}/props", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                info = ModelInfo(
                    name=data.get("default_generation_settings", {}).get("model", "llama.cpp-model")
                )
                info.supports_vision = False  # llama.cpp 原生 server 不支持多模态
                info.supports_tools = False   # 原生 HTTP 接口不支持工具调用
                models.append(info)
        except Exception as e:
            print(f"[LlamaCppBackend] 获取模型信息失败: {e}")
            # 添加一个占位模型
            models.append(ModelInfo(name="llama.cpp-model", supports_vision=False, supports_tools=False))
        return models

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转为 llama.cpp 的 chat completion 格式"""
        result = []
        for msg in messages:
            item = {"role": msg.role, "content": msg.content}
            if msg.images:
                # llama.cpp 支持多模态时的图片格式
                item["images"] = msg.images
            result.append(item)
        return result

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """将消息列表拼接为纯文本 prompt（用于 /completion 接口）"""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                parts.append(f"Tool Result: {msg.content}")
        parts.append("Assistant: ")
        return "\n\n".join(parts)

    def chat(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        # llama.cpp HTTP server 使用 /completion 或 /v1/chat/completions
        # 优先尝试 chat completions 格式
        chat_messages = self._convert_messages(messages)
        payload = {
            "messages": chat_messages,
            "temperature": temperature,
            "stream": True,
        }
        # 如果有工具，尝试以 system prompt 方式注入
        if tools:
            tool_desc = self._tools_to_prompt(tools)
            if chat_messages and chat_messages[0]["role"] == "system":
                chat_messages[0]["content"] += "\n\n" + tool_desc
            else:
                chat_messages.insert(0, {"role": "system", "content": tool_desc})
            payload["messages"] = chat_messages

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            )
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            # 降级到 /completion 接口
            yield from self._fallback_completion(messages, temperature, **kwargs)

    def _fallback_completion(
        self, messages: List[ChatMessage], temperature: float, **kwargs
    ) -> Generator[str, None, None]:
        """使用 /completion 接口作为降级方案"""
        prompt = self._build_prompt(messages)
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                stream=True,
                timeout=300,
            )
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
                    else:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield f"\n[错误] llama.cpp 连接失败: {e}\n"

    def chat_complete(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        chat_messages = self._convert_messages(messages)
        payload = {
            "messages": chat_messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            tool_desc = self._tools_to_prompt(tools)
            if chat_messages and chat_messages[0]["role"] == "system":
                chat_messages[0]["content"] += "\n\n" + tool_desc
            else:
                chat_messages.insert(0, {"role": "system", "content": tool_desc})
            payload["messages"] = chat_messages

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _tools_to_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """将工具定义转为 prompt 文本（llama.cpp 原生不支持工具调用时的降级方案）"""
        lines = ["你可以使用以下工具:"]
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            lines.append(f"\n工具名: {name}")
            lines.append(f"描述: {desc}")
            if params:
                lines.append(f"参数: {json.dumps(params, ensure_ascii=False)}")
            lines.append(f'调用格式: <tool>{name}</tool> 或 {{"tool": "{name}", "parameters": {{...}}}}')
        lines.append("\n当你需要使用工具时，请按上述格式输出。")
        return "\n".join(lines)
