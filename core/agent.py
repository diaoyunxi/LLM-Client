"""
智能体循环模块
实现多轮对话中的工具调用解析与自动执行
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass, field

from .backend import Backend, ChatMessage
from .conversation import Conversation
from .tools.loader import ToolLoader


logger = logging.getLogger("agent")


@dataclass
class AgentStep:
    """智能体执行步骤"""
    step_type: str  # "think" / "tool_call" / "tool_result" / "respond"
    content: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    tool_error: str = ""


class AgentLoop:
    """
    智能体循环
    管理对话流程，自动检测并执行工具调用
    """

    def __init__(
        self,
        backend: Backend,
        conversation: Conversation,
        tool_loader: ToolLoader,
        model: str = "",
        max_iterations: int = 10,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.conversation = conversation
        self.tool_loader = tool_loader
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.step_callbacks: List[Callable[[AgentStep], None]] = []

    def add_step_callback(self, callback: Callable[[AgentStep], None]) -> None:
        """添加步骤回调，用于界面更新"""
        self.step_callbacks.append(callback)

    def _notify_step(self, step: AgentStep) -> None:
        """通知所有回调"""
        for cb in self.step_callbacks:
            try:
                cb(step)
            except Exception as e:
                print(f"[AgentLoop] 回调错误: {e}")

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        从模型回复中提取工具调用
        支持多种格式：
        1. Ollama 原生格式（已在外部解析）
        2. JSON 格式: {"tool": "name", "parameters": {...}}
        3. XML 格式: <tool name="...">...</tool>
        4. Markdown 代码块: ```tool\n{...}\n```
        """
        tool_calls = []

        # 尝试匹配 Markdown 代码块中的 JSON
        code_block_pattern = r'```(?:json|tool)?\s*\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                if "tool" in data or "name" in data:
                    tool_calls.append(self._normalize_tool_call(data))
            except json.JSONDecodeError:
                pass

        # 尝试匹配内联 JSON 对象
        inline_json_pattern = r'\{\s*"(?:tool|name)"\s*:\s*"[^"]+"[^}]*\}'
        for match in re.finditer(inline_json_pattern, content):
            try:
                data = json.loads(match.group(0))
                tool_calls.append(self._normalize_tool_call(data))
            except json.JSONDecodeError:
                pass

        # 尝试匹配 XML 格式
        xml_pattern = r'<tool\s+name="([^"]+)"[^>]*>(.*?)</tool>'
        for match in re.finditer(xml_pattern, content, re.DOTALL):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2).strip())
            except Exception:
                args = {"content": match.group(2).strip()}
            tool_calls.append({"name": tool_name, "arguments": args})

        return tool_calls

    def _normalize_tool_call(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """规范化工具调用格式"""
        name = data.get("tool") or data.get("name") or data.get("function", {}).get("name", "")
        args = data.get("parameters") or data.get("arguments") or data.get("params", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"input": args}
        return {"name": name, "arguments": args}

    def _has_tool_calls(self, content: str) -> bool:
        """检查回复中是否包含工具调用"""
        return len(self._extract_tool_calls(content)) > 0

    def run(
        self,
        user_input: str,
        images: List[str] = None,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        运行智能体循环
        如果 stream=True，逐字产生回复内容
        如果检测到工具调用，自动执行并继续对话
        """
        # 添加用户消息
        self.conversation.add_message("user", user_input, images=images or [])

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1

            # 获取上下文
            messages = self.conversation.get_context_messages()
            tools = self.tool_loader.get_tool_definitions()

            # 调用模型: 统一使用流式或非流式, 不在流式输出后重新请求
            if stream:
                # 流式输出: 逐字产生回复, 仅从文本中提取工具调用, 不重新请求
                full_response = ""
                for chunk in self.backend.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    stream=True,
                    temperature=self.temperature,
                ):
                    full_response += chunk
                    yield chunk

                # 从流式输出文本中提取工具调用 (不再发起非流式重请求)
                tool_calls = self._extract_tool_calls(full_response)

                self.conversation.add_message("assistant", full_response)

                if not tool_calls:
                    # 没有工具调用，对话结束
                    break

                # 执行工具调用
                for tc in tool_calls:
                    yield f"\n[工具调用] {tc['name']}: {json.dumps(tc['arguments'], ensure_ascii=False)}\n"
                    self._notify_step(AgentStep(
                        step_type="tool_call",
                        tool_name=tc["name"],
                        tool_args=tc["arguments"],
                    ))

                    result = self.tool_loader.execute(tc["name"], tc["arguments"])

                    if result["success"]:
                        result_text = json.dumps(result["output"], ensure_ascii=False) if not isinstance(result["output"], str) else result["output"]
                        yield f"[工具结果] {result_text}\n"
                        self._notify_step(AgentStep(
                            step_type="tool_result",
                            tool_name=tc["name"],
                            tool_result=result["output"],
                        ))
                    else:
                        yield f"[工具错误] {result['error']}\n"
                        self._notify_step(AgentStep(
                            step_type="tool_result",
                            tool_name=tc["name"],
                            tool_error=result["error"],
                        ))

                    # 添加工具结果到对话
                    self.conversation.add_message(
                        "tool",
                        json.dumps(result, ensure_ascii=False),
                        **{"tool_call_id": tc.get("id", ""), "name": tc["name"]}
                    )

            else:
                # 非流式（后续迭代或不需要流式）
                response = self.backend.chat_complete(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=self.temperature,
                )

                if "error" in response:
                    error_msg = f"[错误] {response['error']}"
                    self.conversation.add_message("assistant", error_msg)
                    yield error_msg
                    break

                msg = response.get("message", {})
                content = msg.get("content", "")
                native_tool_calls = msg.get("tool_calls", [])

                self.conversation.add_message("assistant", content)

                if not native_tool_calls and not self._has_tool_calls(content):
                    yield content
                    break

                # 提取工具调用
                tool_calls = []
                for tc in native_tool_calls:
                    func = tc.get("function", {})
                    tool_calls.append({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    })
                if not tool_calls:
                    tool_calls = self._extract_tool_calls(content)

                for tc in tool_calls:
                    yield f"\n[工具调用] {tc['name']}\n"
                    result = self.tool_loader.execute(tc["name"], tc["arguments"])
                    result_text = json.dumps(result, ensure_ascii=False)
                    yield f"[工具结果] {result_text}\n"

                    self.conversation.add_message(
                        "tool",
                        result_text,
                        **{"tool_call_id": tc.get("id", ""), "name": tc["name"]}
                    )

        if iteration >= self.max_iterations:
            yield "\n[系统] 已达到最大迭代次数，对话终止。\n"
