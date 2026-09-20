"""
智能体循环模块单元测试

覆盖:
- _extract_tool_calls: Markdown 代码块 / 内联 JSON / XML 格式解析
- _normalize_tool_call: 工具调用格式规范化
- _has_tool_calls: 工具调用检测
"""

import sys
import os
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# 由于 AgentLoop 依赖 Backend/Conversation/ToolLoader, 使用 Mock 对象构造
# ---------------------------------------------------------------------------
class MockBackend:
    """模拟后端"""
    def __init__(self):
        self.host = "localhost"
        self.port = 11434

    def chat(self, **kwargs):
        return iter([])

    def chat_complete(self, **kwargs):
        return {"message": {"content": ""}}


class MockConversation:
    """模拟对话"""
    def __init__(self):
        self.messages = []

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content})

    def get_context_messages(self):
        return []

    def clear(self):
        self.messages.clear()


class MockToolLoader:
    """模拟工具加载器"""
    def get_tool_definitions(self):
        return []

    def execute(self, name, arguments):
        return {"success": True, "output": "mock_result", "error": None}


def _make_agent():
    """构造测试用 AgentLoop 实例"""
    from core.agent import AgentLoop
    return AgentLoop(
        backend=MockBackend(),
        conversation=MockConversation(),
        tool_loader=MockToolLoader(),
        model="test-model",
    )


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------
class TestExtractToolCalls:
    """工具调用提取测试"""

    def test_markdown_code_block_json(self):
        """从 Markdown 代码块中提取 JSON 工具调用"""
        agent = _make_agent()
        content = '一些文字\n```json\n{"tool": "calculator", "parameters": {"expression": "1+2"}}\n```\n更多文字'
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        assert calls[0]["arguments"]["expression"] == "1+2"

    def test_markdown_code_block_no_tool_keyword(self):
        """JSON 中无 tool/name 字段时不应提取"""
        agent = _make_agent()
        content = '```json\n{"key": "value"}\n```'
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 0

    def test_inline_json_not_supported(self):
        """普通文本中的内联 JSON (非代码块) 不应被提取, 避免误判"""
        agent = _make_agent()
        content = '请调用 {"tool": "web_search", "parameters": {"query": "test"}} 搜索'
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 0

    def test_xml_format(self):
        """XML 格式工具调用"""
        agent = _make_agent()
        content = '<tool name="calculator">{"expression": "2+3"}</tool>'
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"

    def test_xml_format_non_json_args(self):
        """XML 格式中非 JSON 内容应作为 content 参数"""
        agent = _make_agent()
        content = '<tool name="echo">hello world</tool>'
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["name"] == "echo"
        assert calls[0]["arguments"]["content"] == "hello world"

    def test_no_tool_calls(self):
        """普通文本无工具调用"""
        agent = _make_agent()
        content = "这是一段普通的回复，没有工具调用。"
        calls = agent._extract_tool_calls(content)
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# _normalize_tool_call
# ---------------------------------------------------------------------------
class TestNormalizeToolCall:
    """工具调用规范化测试"""

    def test_tool_key(self):
        agent = _make_agent()
        result = agent._normalize_tool_call({"tool": "calc", "parameters": {"x": 1}})
        assert result["name"] == "calc"
        assert result["arguments"] == {"x": 1}

    def test_name_key(self):
        agent = _make_agent()
        result = agent._normalize_tool_call({"name": "search", "arguments": {"q": "test"}})
        assert result["name"] == "search"
        assert result["arguments"] == {"q": "test"}

    def test_function_nested(self):
        agent = _make_agent()
        result = agent._normalize_tool_call({
            "function": {"name": "nested_tool"},
            "params": {"a": 1}
        })
        assert result["name"] == "nested_tool"
        assert result["arguments"] == {"a": 1}

    def test_string_args(self):
        """字符串参数应尝试 JSON 解析, 失败则包装为 input"""
        agent = _make_agent()
        result = agent._normalize_tool_call({"tool": "echo", "parameters": '{"msg": "hi"}'})
        assert result["arguments"] == {"msg": "hi"}

    def test_string_args_invalid_json(self):
        agent = _make_agent()
        result = agent._normalize_tool_call({"tool": "echo", "parameters": "plain text"})
        assert result["arguments"] == {"input": "plain text"}


# ---------------------------------------------------------------------------
# _has_tool_calls
# ---------------------------------------------------------------------------
class TestHasToolCalls:
    """工具调用检测测试"""

    def test_has_calls(self):
        agent = _make_agent()
        content = '```json\n{"tool": "calc", "parameters": {}}\n```'
        assert agent._has_tool_calls(content) is True

    def test_no_calls(self):
        agent = _make_agent()
        content = "普通文本"
        assert agent._has_tool_calls(content) is False


# ---------------------------------------------------------------------------
# _notify_step
# ---------------------------------------------------------------------------
class TestNotifyStep:
    """步骤回调测试"""

    def test_callback_invoked(self):
        from core.agent import AgentStep
        agent = _make_agent()
        received = []
        agent.add_step_callback(lambda step: received.append(step))
        agent._notify_step(AgentStep(step_type="think", content="思考中"))
        assert len(received) == 1
        assert received[0].step_type == "think"

    def test_callback_error_handled(self):
        """回调异常不应中断主流程"""
        from core.agent import AgentStep
        agent = _make_agent()

        def bad_callback(step):
            raise RuntimeError("回调爆炸")

        agent.add_step_callback(bad_callback)
        # 不应抛出异常
        agent._notify_step(AgentStep(step_type="think"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
