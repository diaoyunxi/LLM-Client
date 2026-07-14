"""
工具加载器
扫描目录、加载外置工具、执行工具调用
"""

import os
import sys
import json
import importlib.util
from typing import Dict, List, Any, Optional, Callable
from .parser import ToolDefinition, parse_tool_from_file


class ToolLoader:
    """工具加载器"""

    def __init__(self, tools_dirs: List[str] = None):
        self.tools_dirs = tools_dirs or []
        self.tools: Dict[str, ToolDefinition] = {}
        self._functions: Dict[str, Callable] = {}
        self._modules: Dict[str, Any] = {}

    def add_tools_dir(self, directory: str) -> None:
        """添加工具目录"""
        if os.path.isdir(directory) and directory not in self.tools_dirs:
            self.tools_dirs.append(directory)

    def load_all(self) -> int:
        """
        扫描所有工具目录，加载工具
        返回加载的工具数量
        """
        loaded = 0
        for directory in self.tools_dirs:
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                if not filename.endswith('.py') or filename.startswith('_'):
                    continue
                filepath = os.path.join(directory, filename)
                if self.load_tool(filepath):
                    loaded += 1
        return loaded

    def load_tool(self, filepath: str) -> bool:
        """
        加载单个工具文件
        """
        tool_def = parse_tool_from_file(filepath)
        if not tool_def:
            return False

        # 动态导入模块
        module_name = f"tool_{tool_def.name}_{hash(filepath) % 10000}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._modules[tool_def.name] = module

            # 获取执行函数
            func_name = tool_def.function_name
            if hasattr(module, func_name):
                self._functions[tool_def.name] = getattr(module, func_name)
            else:
                print(f"[ToolLoader] 警告: 工具 {tool_def.name} 未找到函数 {func_name}")
                self._functions[tool_def.name] = None

            self.tools[tool_def.name] = tool_def
            print(f"[ToolLoader] 已加载工具: {tool_def.name}")
            return True

        except Exception as e:
            print(f"[ToolLoader] 加载工具失败 {filepath}: {e}")
            return False

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有工具的 Ollama 格式定义"""
        return [tool.to_ollama_format() for tool in self.tools.values()]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self.tools.get(name)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具调用
        返回标准格式的结果字典
        """
        tool = self.tools.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"工具 '{name}' 未找到",
                "output": None,
            }

        # 参数验证
        valid, error = tool.validate_args(arguments)
        if not valid:
            return {
                "success": False,
                "error": f"参数验证失败: {error}",
                "output": None,
            }

        func = self._functions.get(name)
        if not func:
            return {
                "success": False,
                "error": f"工具 '{name}' 的执行函数未加载",
                "output": None,
            }

        try:
            result = func(**arguments)
            return {
                "success": True,
                "error": None,
                "output": result,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None,
            }

    def unload_tool(self, name: str) -> bool:
        """卸载工具"""
        if name in self.tools:
            del self.tools[name]
            del self._functions[name]
            if name in self._modules:
                del self._modules[name]
            return True
        return False

    def reload_tool(self, filepath: str) -> bool:
        """重新加载工具"""
        tool_def = parse_tool_from_file(filepath)
        if tool_def and tool_def.name in self.tools:
            self.unload_tool(tool_def.name)
        return self.load_tool(filepath)
