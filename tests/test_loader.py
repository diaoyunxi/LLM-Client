"""
工具加载器单元测试

覆盖:
- load_tool: 从文件加载工具定义与执行函数
- load_all: 批量加载目录下工具
- execute: 工具调用执行与参数验证
- unload_tool: 工具卸载 (使用 .pop)
- get_tool / get_tool_definitions: 工具查询
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.tools.loader import ToolLoader


# ---------------------------------------------------------------------------
# 测试用工具文件内容
# ---------------------------------------------------------------------------
TOOL_FILE_CONTENT = '''"""
TOOL_NAME: test_tool
TOOL_DESCRIPTION: 测试工具
TOOL_PARAMETERS:
    message:
        type: string
        description: 测试消息
        required: true
"""

def run(message: str):
    return f"收到: {message}"
'''

TOOL_FILE_CONTENT_OPTIONAL = '''"""
TOOL_NAME: optional_tool
TOOL_DESCRIPTION: 带可选参数的测试工具
TOOL_PARAMETERS:
    name:
        type: string
        description: 名称
        required: true
    count:
        type: integer
        description: 数量
        required: false
        default: 1
"""

def run(name: str, count: int = 1):
    return {"name": name, "count": count}
'''


@pytest.fixture
def temp_tools_dir():
    """创建临时工具目录, 测试结束后清理"""
    d = tempfile.mkdtemp(prefix="test_tools_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def loaded_loader(temp_tools_dir):
    """创建并加载工具的 ToolLoader"""
    # 写入测试工具文件
    with open(os.path.join(temp_tools_dir, "test_tool.py"), "w", encoding="utf-8") as f:
        f.write(TOOL_FILE_CONTENT)
    with open(os.path.join(temp_tools_dir, "optional_tool.py"), "w", encoding="utf-8") as f:
        f.write(TOOL_FILE_CONTENT_OPTIONAL)

    loader = ToolLoader()
    loader.add_tools_dir(temp_tools_dir)
    loader.load_all()
    return loader


# ---------------------------------------------------------------------------
# load_tool
# ---------------------------------------------------------------------------
class TestLoadTool:
    """工具加载测试"""

    def test_load_single_tool(self, temp_tools_dir):
        filepath = os.path.join(temp_tools_dir, "test_tool.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(TOOL_FILE_CONTENT)

        loader = ToolLoader()
        assert loader.load_tool(filepath) is True
        assert "test_tool" in loader.tools

    def test_load_nonexistent_file(self):
        loader = ToolLoader()
        assert loader.load_tool("/nonexistent/path/tool.py") is False

    def test_load_all(self, loaded_loader):
        """批量加载"""
        assert "test_tool" in loaded_loader.tools
        assert "optional_tool" in loaded_loader.tools

    def test_skip_underscore_files(self, temp_tools_dir):
        """以下划线开头的文件应被跳过"""
        with open(os.path.join(temp_tools_dir, "_private.py"), "w", encoding="utf-8") as f:
            f.write(TOOL_FILE_CONTENT)
        with open(os.path.join(temp_tools_dir, "test_tool.py"), "w", encoding="utf-8") as f:
            f.write(TOOL_FILE_CONTENT)

        loader = ToolLoader()
        loader.add_tools_dir(temp_tools_dir)
        count = loader.load_all()
        assert count == 1
        assert "_private" not in loader.tools


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------
class TestExecute:
    """工具执行测试"""

    def test_execute_success(self, loaded_loader):
        result = loaded_loader.execute("test_tool", {"message": "hello"})
        assert result["success"] is True
        assert result["output"] == "收到: hello"
        assert result["error"] is None

    def test_execute_not_found(self, loaded_loader):
        result = loaded_loader.execute("nonexistent", {})
        assert result["success"] is False
        assert "未找到" in result["error"]

    def test_execute_missing_required_arg(self, loaded_loader):
        result = loaded_loader.execute("test_tool", {})
        assert result["success"] is False
        assert "缺少必需参数" in result["error"]

    def test_execute_wrong_type(self, loaded_loader):
        result = loaded_loader.execute("test_tool", {"message": 123})
        assert result["success"] is False
        assert "应为字符串" in result["error"]

    def test_execute_with_optional_arg(self, loaded_loader):
        result = loaded_loader.execute("optional_tool", {"name": "test", "count": 5})
        assert result["success"] is True
        assert result["output"]["count"] == 5

    def test_execute_optional_default(self, loaded_loader):
        result = loaded_loader.execute("optional_tool", {"name": "test"})
        assert result["success"] is True
        assert result["output"]["count"] == 1


# ---------------------------------------------------------------------------
# unload_tool
# ---------------------------------------------------------------------------
class TestUnloadTool:
    """工具卸载测试"""

    def test_unload_existing(self, loaded_loader):
        assert "test_tool" in loaded_loader.tools
        assert loaded_loader.unload_tool("test_tool") is True
        assert "test_tool" not in loaded_loader.tools
        assert "test_tool" not in loaded_loader._functions

    def test_unload_nonexistent(self, loaded_loader):
        """卸载不存在的工具应返回 False, 不抛异常"""
        assert loaded_loader.unload_tool("nonexistent") is False

    def test_unload_then_reload(self, loaded_loader, temp_tools_dir):
        filepath = os.path.join(temp_tools_dir, "test_tool.py")
        loaded_loader.unload_tool("test_tool")
        assert "test_tool" not in loaded_loader.tools
        assert loaded_loader.load_tool(filepath) is True
        assert "test_tool" in loaded_loader.tools


# ---------------------------------------------------------------------------
# get_tool / get_tool_definitions
# ---------------------------------------------------------------------------
class TestGetTool:
    """工具查询测试"""

    def test_get_tool(self, loaded_loader):
        tool = loaded_loader.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "测试工具"

    def test_get_tool_not_found(self, loaded_loader):
        assert loaded_loader.get_tool("nonexistent") is None

    def test_get_tool_definitions(self, loaded_loader):
        defs = loaded_loader.get_tool_definitions()
        assert len(defs) == 2
        names = {d["function"]["name"] for d in defs}
        assert "test_tool" in names
        assert "optional_tool" in names

    def test_tool_definition_format(self, loaded_loader):
        defs = loaded_loader.get_tool_definitions()
        for d in defs:
            assert d["type"] == "function"
            assert "function" in d
            assert "name" in d["function"]
            assert "parameters" in d["function"]


# ---------------------------------------------------------------------------
# reload_tool
# ---------------------------------------------------------------------------
class TestReloadTool:
    """工具重载测试"""

    def test_reload(self, loaded_loader, temp_tools_dir):
        filepath = os.path.join(temp_tools_dir, "test_tool.py")
        # 修改工具文件
        new_content = TOOL_FILE_CONTENT.replace("收到:", "已接收:")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        assert loaded_loader.reload_tool(filepath) is True
        result = loaded_loader.execute("test_tool", {"message": "hi"})
        assert result["output"] == "已接收: hi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
