"""
工具加载器
扫描目录、加载外置工具、执行工具调用
"""

import logging
import os
import sys
import json
import uuid
import importlib.util
from typing import Dict, List, Any, Optional, Callable
from .parser import ToolDefinition, parse_tool_from_file


logger = logging.getLogger("tool_loader")


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

        # 动态导入模块 (使用 uuid 生成唯一模块名, 避免 hash() 跨进程不一致)
        module_name = f"tool_{tool_def.name}_{uuid.uuid4().hex[:8]}"
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
                logger.warning("工具 %s 未找到函数 %s", tool_def.name, func_name)
                self._functions[tool_def.name] = None

            self.tools[tool_def.name] = tool_def
            logger.info("已加载工具: %s", tool_def.name)
            return True

        except Exception as e:
            logger.error("加载工具失败 %s: %s", filepath, e)
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
                "error": f"{type(e).__name__}",
                "output": None,
            }

    def unload_tool(self, name: str) -> bool:
        """卸载工具

        同时清理 sys.modules 中对应的动态模块条目，
        防止每次加载/卸载遗留孤儿模块引用导致内存泄漏。
        """
        if name in self.tools:
            # 先记录模块名以便清理 sys.modules
            old_module = self._modules.get(name)
            old_module_name = getattr(old_module, '__name__', None) if old_module else None
            self.tools.pop(name, None)
            self._functions.pop(name, None)
            self._modules.pop(name, None)
            # 清理 sys.modules 中的孤儿模块条目
            if old_module_name and old_module_name in sys.modules:
                del sys.modules[old_module_name]
            return True
        return False

    def reload_tool(self, filepath: str) -> bool:
        """重新加载工具"""
        tool_def = parse_tool_from_file(filepath)
        if tool_def and tool_def.name in self.tools:
            self.unload_tool(tool_def.name)
        return self.load_tool(filepath)
