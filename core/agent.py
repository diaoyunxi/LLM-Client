"""
智能体循环模块
实现多轮对话中的工具调用解析与自动执行
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass, field

from .backend import Backend, ChatMessage, StreamChunk
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
        think: bool = False,
    ):
        self.backend = backend
        self.conversation = conversation
        self.tool_loader = tool_loader
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.think = think
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

    def _stream_chat_iteration(
        self,
        messages: list,
        tools: list | None,
    ) -> tuple[str, str, list[dict]]:
        """Execute one streaming iteration, yielding chunks and returning full response.

        Returns:
            (full_response, full_thinking, tool_calls)
        """
        full_response = ""
        full_thinking = ""
        in_thinking = False

        for chunk in self.backend.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            temperature=self.temperature,
            think=self.think,
        ):
            if chunk.thinking:
                full_thinking += chunk.thinking
                in_thinking = True
                yield StreamChunk(thinking=chunk.thinking)
            if chunk.content:
                full_response += chunk.content
                in_thinking = False
                yield StreamChunk(content=chunk.content)

        tool_calls = self._extract_tool_calls(full_response)
        return full_response, full_thinking, tool_calls

    def _execute_tool_calls_streaming(
        self,
        tool_calls: list[dict],
    ) -> Generator[StreamChunk, None, None]:
        """Execute tool calls and yield results as StreamChunks.

        Also notifies step callbacks and adds results to conversation.
        """
        for tc in tool_calls:
            tool_text = f"\n[工具调用] {tc['name']}: {json.dumps(tc['arguments'], ensure_ascii=False)}\n"
            yield StreamChunk(content=tool_text)
            self._notify_step(AgentStep(
                step_type="tool_call",
                tool_name=tc["name"],
                tool_args=tc["arguments"],
            ))

            result = self.tool_loader.execute(tc["name"], tc["arguments"])

            if result["success"]:
                result_text = (
                    json.dumps(result["output"], ensure_ascii=False)
                    if not isinstance(result["output"], str)
                    else result["output"]
                )
                yield StreamChunk(content=f"[工具结果] {result_text}\n")
                self._notify_step(AgentStep(
                    step_type="tool_result",
                    tool_name=tc["name"],
                    tool_result=result["output"],
                ))
            else:
                yield StreamChunk(content=f"[工具错误] {result['error']}\n")
                self._notify_step(AgentStep(
                    step_type="tool_result",
                    tool_name=tc["name"],
                    tool_error=result["error"],
                ))

            self.conversation.add_message(
                "tool",
                json.dumps(result, ensure_ascii=False),
                **{"tool_call_id": tc.get("id", ""), "name": tc["name"]}
            )

    def _non_stream_iteration(
        self,
        messages: list,
        tools: list | None,
    ) -> tuple[str, str, list[dict], bool]:
        """Execute one non-streaming iteration.

        Returns:
            (content, thinking, tool_calls, should_break)
        """
        response = self.backend.chat_complete(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            think=self.think,
        )

        if "error" in response:
            error_msg = f"[错误] {response['error']}"
            self.conversation.add_message("assistant", error_msg)
            return error_msg, "", [], True

        msg = response.get("message", {})
        content = msg.get("content", "")
        thinking_content = msg.get("thinking", "")
        native_tool_calls = msg.get("tool_calls", [])

        self.conversation.add_message("assistant", content, thinking=thinking_content)

        if thinking_content:
            yield StreamChunk(thinking=thinking_content)

        # Extract tool calls from native format or text
        tool_calls = []
        for tc in native_tool_calls:
            func = tc.get("function", {})
            tool_calls.append({
                "name": func.get("name", ""),
                "arguments": func.get("arguments", {}),
            })
        if not tool_calls:
            tool_calls = self._extract_tool_calls(content)

        should_break = not native_tool_calls and not self._has_tool_calls(content)
        if should_break:
            yield StreamChunk(content=content)

        return content, thinking_content, tool_calls, should_break

    def run(
        self,
        user_input: str,
        images: list[str] | None = None,
        stream: bool = True,
    ) -> Generator[StreamChunk, None, None]:
        """Run the agent loop, yielding StreamChunks for thinking and content.

        Automatically detects and executes tool calls, continuing the conversation
        until no more tool calls are detected or max_iterations is reached.
        """
        self.conversation.add_message("user", user_input, images=images or [])

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            messages = self.conversation.get_context_messages()
            tools = self.tool_loader.get_tool_definitions() or None

            if stream:
                yield from self._run_streaming(messages, tools)
                break  # Streaming handles its own iteration logic
            else:
                should_continue = yield from self._run_non_streaming(messages, tools)
                if not should_continue:
                    break

        if iteration >= self.max_iterations:
            yield StreamChunk(content="\n[系统] 已达到最大迭代次数，对话终止。\n")

    def _run_streaming(
        self,
        messages: list,
        tools: list | None,
    ) -> Generator[StreamChunk, None, None]:
        """Handle streaming mode iteration with tool call detection and execution."""
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            messages = self.conversation.get_context_messages()
            tools = self.tool_loader.get_tool_definitions() or None

            full_response, full_thinking, tool_calls = yield from self._stream_chat_iteration(
                messages, tools
            )
            self.conversation.add_message("assistant", full_response, thinking=full_thinking)

            if not tool_calls:
                break

            yield from self._execute_tool_calls_streaming(tool_calls)

    def _run_non_streaming(
        self,
        messages: list,
        tools: list | None,
    ) -> Generator[StreamChunk, None, bool]:
        """Handle non-streaming mode iteration.

        Returns:
            True if should continue iterating, False if done.
        """
        content, thinking, tool_calls, should_break = yield from self._non_stream_iteration(
            messages, tools
        )

        if should_break:
            return False

        for tc in tool_calls:
            yield StreamChunk(content=f"\n[工具调用] {tc['name']}\n")
            result = self.tool_loader.execute(tc["name"], tc["arguments"])
            result_text = json.dumps(result, ensure_ascii=False)
            yield StreamChunk(content=f"[工具结果] {result_text}\n")

            self.conversation.add_message(
                "tool",
                result_text,
                **{"tool_call_id": tc.get("id", ""), "name": tc["name"]}
            )

        return True
