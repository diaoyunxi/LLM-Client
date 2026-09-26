"""
后端抽象层
支持 Ollama (通过 ollama 库) 和 llama.cpp (直接 HTTP 连接)
"""

import base64
import json
import os
import logging
import uuid
import requests
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any
from dataclasses import dataclass, field
import ollama


logger = logging.getLogger("backend")

# 通用错误消息, 避免向后端用户泄露内部异常细节
GENERIC_ERROR_MSG = "后端服务暂时不可用"


@dataclass
class StreamChunk:
    """流式输出块，区分思考内容和正式回复"""
    thinking: str = ""   # 思考过程内容（可能为空）
    content: str = ""    # 正式回复内容


@dataclass
class ChatMessage:
    """统一消息格式"""
    role: str  # system / user / assistant / tool
    content: str = ""
    images: List[str] = field(default_factory=list)  # base64 编码的图片
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    thinking: str = ""   # 模型思考过程（仅 assistant 消息）


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

    # 请求重试配置
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # 退避基数 (秒), 第 n 次重试等待 base * 2^n

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def _request_with_retry(self, func, *args, **kwargs):
        """
        带指数退避的请求重试机制

        最多重试 MAX_RETRIES 次, 每次重试前等待 RETRY_BACKOFF_BASE * 2^(attempt-1) 秒。
        仅对网络/连接异常重试, 业务逻辑错误不重试。

        Args:
            func: 可调用对象 (如 self.client.chat 或 requests.post)
            *args, **kwargs: 传递给 func 的参数

        Returns:
            func 的返回值

        Raises:
            最后一次重试仍失败时抛出原始异常
        """
        import time
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except (requests.ConnectionError, requests.Timeout, OSError) as e:
                last_exc = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning("请求失败 (第 %d 次), %.1fs 后重试: %s", attempt, delay, e)
                    time.sleep(delay)
                else:
                    logger.warning("请求失败, 已达最大重试次数 %d: %s", self.MAX_RETRIES, e)
            except Exception as e:
                # 非网络异常, 不重试直接抛出
                raise
        raise last_exc

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
        think: bool = False,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """
        发起对话
        返回生成器，逐字输出 StreamChunk（区分 thinking 和 content）
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
        """
        将图片转为 base64

        安全限制: 仅允许读取用户主目录下的文件, 防止越权读取系统敏感文件。
        """
        # 路径校验: 限制只能读取用户目录下的文件
        home_dir = os.path.expanduser("~")
        real_path = os.path.realpath(image_path)
        if not real_path.startswith(home_dir):
            raise PermissionError(f"安全限制: 仅允许读取用户目录 ({home_dir}) 下的文件")
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
            # 异常详情记录到日志, 不直接暴露给用户
            logger.warning("Ollama 获取模型列表失败: %s", e)
        return models


    def _detect_image_mime_type(self, img_base64: str) -> str:
        """从 base64 编码的图片数据中检测 MIME 类型"""
        try:
            header_bytes = base64.b64decode(img_base64[:8])
            # PNG: 89 50 4E 47
            if header_bytes.startswith(b'\x89PNG'):
                return "image/png"
            # GIF: 47 49 46 38
            elif header_bytes.startswith(b'GIF8'):
                return "image/gif"
            # WebP: 52 49 46 46 ... 57 45 42 50
            elif header_bytes.startswith(b'RIFF') and len(header_bytes) >= 8:
                full_header = base64.b64decode(img_base64[:24])
                if full_header[8:12] == b'WEBP':
                    return "image/webp"
            # JPEG: FF D8 FF
            elif header_bytes.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            # 默认返回 jpeg
            return "image/jpeg"
        except Exception:
            return "image/jpeg"

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
        think: bool = False,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
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
                think=think,
            )
            for chunk in response:
                msg = chunk.get("message", {})
                thinking_text = msg.get("thinking", "") or ""
                content_text = msg.get("content", "") or ""
                if thinking_text or content_text:
                    yield StreamChunk(thinking=thinking_text, content=content_text)
        except Exception as e:
            # 异常详情记录到日志, 向用户返回通用错误消息
            logger.warning("Ollama 对话失败: %s", e)
            yield StreamChunk(content=f"\n[错误] {GENERIC_ERROR_MSG}\n")

    def chat_complete(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        think: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        ollama_messages = self._convert_messages(messages)
        options = {"temperature": temperature}
        options.update(kwargs)

        try:
            response = self._request_with_retry(
                self.client.chat,
                model=model,
                messages=ollama_messages,
                tools=tools,
                stream=False,
                options=options,
                think=think,
            )
            return response
        except Exception as e:
            # 异常详情记录到日志, 向用户返回通用错误消息
            logger.warning("Ollama 请求失败: %s", e)
            return {"error": GENERIC_ERROR_MSG, "message": {"content": f"[错误] {GENERIC_ERROR_MSG}"}}


class OpenAIBackend(Backend):
    """
    OpenAI API 兼容后端
    支持任何遵循 OpenAI 协议的远程 API 服务（如 OpenAI、vLLM、LocalAI 等）
    """

    def __init__(self, base_url: str = "https://api.openai.com/v1", api_key: str = "", default_model: str = ""):
        # 不调用父类 __init__，因为 URL 格式不同
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self._headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def list_models(self) -> List[ModelInfo]:
        """获取可用模型列表"""
        models = []
        try:
            def _fetch_models():
                return requests.get(f"{self.base_url}/models", headers=self._headers, timeout=10)
            resp = self._request_with_retry(_fetch_models)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    info = ModelInfo(name=m.get("id", "unknown"))
                    # OpenAI 协议通常不直接提供视觉/工具支持信息，这里做简单判断
                    name_lower = info.name.lower()
                    vision_keywords = ["gpt-4-vision", "gpt-4o", "llava", "vision", "multimodal"]
                    info.supports_vision = any(k in name_lower for k in vision_keywords)
                    # 大多数现代模型都支持工具调用
                    info.supports_tools = not any(k in name_lower for k in ["embedding", "tts", "whisper"])
                    models.append(info)
        except Exception as e:
            logger.warning("OpenAI API 获取模型列表失败：%s", e)
            # 如果无法获取，返回默认模型（如果有）
            if self.default_model:
                models.append(ModelInfo(name=self.default_model, supports_vision=True, supports_tools=True))
        return models


    def _detect_image_mime_type(self, img_base64: str) -> str:
        """从 base64 编码的图片数据中检测 MIME 类型"""
        try:
            header_bytes = base64.b64decode(img_base64[:8])
            # PNG: 89 50 4E 47
            if header_bytes.startswith(b'\x89PNG'):
                return "image/png"
            # GIF: 47 49 46 38
            elif header_bytes.startswith(b'GIF8'):
                return "image/gif"
            # WebP: 52 49 46 46 ... 57 45 42 50
            elif header_bytes.startswith(b'RIFF') and len(header_bytes) >= 8:
                full_header = base64.b64decode(img_base64[:24])
                if full_header[8:12] == b'WEBP':
                    return "image/webp"
            # JPEG: FF D8 FF
            elif header_bytes.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            # 默认返回 jpeg
            return "image/jpeg"
        except Exception:
            return "image/jpeg"

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """将内部消息格式转为 OpenAI 格式"""
        result = []
        for msg in messages:
            # OpenAI 格式处理多模态内容
            content = msg.content
            if msg.images:
                # 构建多模态内容数组
                content_list = []
                if msg.content:
                    content_list.append({"type": "text", "text": msg.content})
                for img_base64 in msg.images:
                    # 检测实际图片 MIME 类型
                    mime_type = self._detect_image_mime_type(img_base64)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                    })
                content = content_list
            
            item = {"role": msg.role, "content": content}
            
            # 处理 tool_calls (assistant 消息)
            if msg.tool_calls:
                openai_tool_calls = []
                for tc in msg.tool_calls:
                    openai_tool_calls.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": json.dumps(tc.get("function", {}).get("arguments", {}))
                        }
                    })
                item["tool_calls"] = openai_tool_calls
            
            # 处理 tool_call_id (tool 消息)
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
        think: bool = False,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        openai_messages = self._convert_messages(messages)
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
        }
        
        # 添加工具定义（OpenAI 格式）
        if tools:
            payload["tools"] = tools
        
        # 系统提示词已在 messages 中处理
        
        try:
            def _make_request():
                return requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    stream=True,
                    timeout=(10, 60),
                )
            resp = self._request_with_retry(_make_request)
            resp.raise_for_status()
            
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
                            
                            # 处理思考过程（某些模型如 DeepSeek-R1 通过 reasoning_content 返回）
                            thinking_text = delta.get("reasoning_content", "") or ""
                            content_text = delta.get("content", "") or ""
                            
                            # 处理工具调用
                            if "tool_calls" in delta and delta["tool_calls"]:
                                # 工具调用通常在 content 为空时出现，这里不做特殊流式处理
                                pass
                            
                            if thinking_text or content_text:
                                yield StreamChunk(thinking=thinking_text, content=content_text)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning("OpenAI API 对话失败：%s", e)
            yield StreamChunk(content=f"\n[错误] {GENERIC_ERROR_MSG}\n")

    def chat_complete(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        think: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        openai_messages = self._convert_messages(messages)
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": False,
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            def _make_request():
                return requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=(10, 60),
                )
            resp = self._request_with_retry(_make_request)
            resp.raise_for_status()
            result = resp.json()
            # 规范化响应格式以匹配 AgentLoop.run 的期望
            # OpenAI 返回：{"choices": [{"message": {...}}]}
            # AgentLoop 期望：{"message": {...}}
            if "choices" in result and len(result["choices"]) > 0:
                msg = result["choices"][0].get("message", {})
                return {"message": msg}
            return result
        except Exception as e:
            logger.warning("OpenAI API 请求失败：%s", e)
            return {"error": GENERIC_ERROR_MSG, "message": {"content": f"[错误] {GENERIC_ERROR_MSG}"}}


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
            # 异常详情记录到日志, 不直接暴露给用户
            logger.warning("llama.cpp 获取模型信息失败: %s", e)
            # 添加一个占位模型
            models.append(ModelInfo(name="llama.cpp-model", supports_vision=False, supports_tools=False))
        return models


    def _detect_image_mime_type(self, img_base64: str) -> str:
        """从 base64 编码的图片数据中检测 MIME 类型"""
        try:
            header_bytes = base64.b64decode(img_base64[:8])
            # PNG: 89 50 4E 47
            if header_bytes.startswith(b'\x89PNG'):
                return "image/png"
            # GIF: 47 49 46 38
            elif header_bytes.startswith(b'GIF8'):
                return "image/gif"
            # WebP: 52 49 46 46 ... 57 45 42 50
            elif header_bytes.startswith(b'RIFF') and len(header_bytes) >= 8:
                full_header = base64.b64decode(img_base64[:24])
                if full_header[8:12] == b'WEBP':
                    return "image/webp"
            # JPEG: FF D8 FF
            elif header_bytes.startswith(b'\xff\xd8\xff'):
                return "image/jpeg"
            # 默认返回 jpeg
            return "image/jpeg"
        except Exception:
            return "image/jpeg"

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
        think: bool = False,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
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
                timeout=(10, 60),
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
                                yield StreamChunk(content=content)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            # 降级到 /completion 接口
            yield from self._fallback_completion(messages, temperature, **kwargs)

    def _fallback_completion(
        self, messages: List[ChatMessage], temperature: float, **kwargs
    ) -> Generator[StreamChunk, None, None]:
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
                timeout=(10, 60),
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
                                yield StreamChunk(content=content)
                        except json.JSONDecodeError:
                            pass
                    else:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("content", "")
                            if content:
                                yield StreamChunk(content=content)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            # 异常详情记录到日志, 向用户返回通用错误消息
            logger.warning("llama.cpp 连接失败: %s", e)
            yield StreamChunk(content=f"\n[错误] {GENERIC_ERROR_MSG}\n")

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
                timeout=(10, 60),
            )
            return resp.json()
        except Exception as e:
            # 异常详情记录到日志, 向用户返回通用错误消息
            logger.warning("llama.cpp 请求失败: %s", e)
            return {"error": GENERIC_ERROR_MSG}

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
