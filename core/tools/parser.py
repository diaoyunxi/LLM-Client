"""
工具解析器
从 Python 文件开头的注释中解析工具定义

注释格式示例:
"""
"""
TOOL_NAME: calculator
TOOL_DESCRIPTION: 执行基础数学运算
TOOL_PARAMETERS:
    expression:
        type: string
        description: 数学表达式，如 "1 + 2 * 3"
        required: true
"""
import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    param_type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list = field(default_factory=list)

    def to_schema(self) -> Dict[str, Any]:
        schema = {
            "type": self.param_type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, ToolParameter] = field(default_factory=dict)
    module_path: str = ""
    function_name: str = "run"  # 默认调用函数名

    def to_ollama_format(self) -> Dict[str, Any]:
        """转为 Ollama 工具格式"""
        properties = {}
        required = []
        for name, param in self.parameters.items():
            properties[name] = param.to_schema()
            if param.required:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, args: Dict[str, Any]) -> tuple[bool, str]:
        """验证参数"""
        for name, param in self.parameters.items():
            if param.required and name not in args:
                return False, f"缺少必需参数: {name}"
            if name in args:
                value = args[name]
                # 基础类型检查
                if param.param_type == "string" and not isinstance(value, str):
                    return False, f"参数 {name} 应为字符串"
                elif param.param_type == "integer" and not isinstance(value, int):
                    return False, f"参数 {name} 应为整数"
                elif param.param_type == "number" and not isinstance(value, (int, float)):
                    return False, f"参数 {name} 应为数字"
                elif param.param_type == "boolean" and not isinstance(value, bool):
                    return False, f"参数 {name} 应为布尔值"
                elif param.param_type == "array" and not isinstance(value, list):
                    return False, f"参数 {name} 应为数组"
                elif param.param_type == "object" and not isinstance(value, dict):
                    return False, f"参数 {name} 应为对象"
        return True, ""


def parse_tool_from_file(filepath: str) -> Optional[ToolDefinition]:
    """
    从 Python 文件解析工具定义
    解析文件开头的多行注释中的 TOOL 元数据
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[ToolParser] 读取文件失败 {filepath}: {e}")
        return None

    # 提取文件开头的多行注释（支持 """ 和 '''）
    comment_match = re.match(r'^(?:\s*#.*\n)*\s*("""|\'\'\')(.*?)\1', content, re.DOTALL)
    if not comment_match:
        # 尝试匹配单行注释格式
        comment_match = re.match(r'^(?:\s*#.*\n)*', content)
        if comment_match:
            comment_text = comment_match.group(0)
        else:
            return None
    else:
        comment_text = comment_match.group(2)

    # 解析 TOOL 元数据
    name_match = re.search(r'TOOL_NAME:\s*(\S+)', comment_text)
    desc_match = re.search(r'TOOL_DESCRIPTION:\s*(.+?)(?=\n\w|$)', comment_text, re.DOTALL)

    if not name_match:
        return None  # 没有 TOOL_NAME 标记，不是工具文件

    tool = ToolDefinition(
        name=name_match.group(1).strip(),
        description=desc_match.group(1).strip() if desc_match else "",
        module_path=filepath,
    )

    # 解析参数定义
    params_match = re.search(r'TOOL_PARAMETERS:\s*(.*?)(?=\n\w|$)', comment_text, re.DOTALL)
    if params_match:
        params_text = params_match.group(1)
        # 解析缩进格式的参数定义
        current_param = None
        current_indent = None

        for line in params_text.split('\n'):
            if not line.strip():
                continue
            # 检测顶层参数名（无缩进或少缩进）
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if indent <= 4 and not line.strip().startswith('-'):
                # 新参数
                param_name = stripped.rstrip(':').strip()
                if param_name:
                    current_param = ToolParameter(name=param_name)
                    tool.parameters[param_name] = current_param
                    current_indent = indent
            elif current_param and indent > current_indent:
                # 参数属性
                prop_match = re.match(r'(\w+):\s*(.+)', stripped)
                if prop_match:
                    prop_name = prop_match.group(1).strip()
                    prop_value = prop_match.group(2).strip()

                    if prop_name == "type":
                        current_param.param_type = prop_value
                    elif prop_name == "description":
                        current_param.description = prop_value
                    elif prop_name == "required":
                        current_param.required = prop_value.lower() in ('true', 'yes', '1')
                    elif prop_name == "default":
                        current_param.default = prop_value
                    elif prop_name == "enum":
                        try:
                            current_param.enum = json.loads(prop_value)
                        except:
                            current_param.enum = [v.strip() for v in prop_value.split(',')]

    return tool
