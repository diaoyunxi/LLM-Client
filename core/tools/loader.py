"""
工具加载器
扫描目录、加载外置工具、执行工具调用
"""

import os
import sys
import json
import uuid
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

    # 工具执行超时时间（秒），防止单个工具阻塞整个智能体循环
    EXECUTE_TIMEOUT = 120

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具调用
        返回标准格式的结果字典

        安全措施：使用线程超时机制，防止工具函数无限阻塞。
        超时时间由 EXECUTE_TIMEOUT 类属性控制（默认 120 秒）。
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

        # 使用线程超时机制执行工具，防止无限阻塞
        import threading
        result_holder = {"result": None, "error": None}

        def _run():
            try:
                result_holder["result"] = func(**arguments)
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.EXECUTE_TIMEOUT)

        if thread.is_alive():
            # 超时：线程仍在运行，不等待直接返回错误
            return {
                "success": False,
                "error": f"工具 '{name}' 执行超时（>{self.EXECUTE_TIMEOUT}秒）",
                "output": None,
            }

        if result_holder["error"] is not None:
            return {
                "success": False,
                "error": str(result_holder["error"]),
                "output": None,
            }

        return {
            "success": True,
            "error": None,
            "output": result_holder["result"],
        }

    def unload_tool(self, name: str) -> bool:
        """卸载工具"""
        if name in self.tools:
            self.tools.pop(name, None)
            self._functions.pop(name, None)
            self._modules.pop(name, None)
            return True
        return False

    def reload_tool(self, filepath: str) -> bool:
        """重新加载工具"""
        tool_def = parse_tool_from_file(filepath)
        if tool_def and tool_def.name in self.tools:
            self.unload_tool(tool_def.name)
        return self.load_tool(filepath)
