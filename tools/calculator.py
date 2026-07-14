"""
TOOL_NAME: calculator
TOOL_DESCRIPTION: 执行基础数学运算，支持加减乘除、幂运算和取模
TOOL_PARAMETERS:
    expression:
        type: string
        description: 数学表达式，如 "1 + 2 * 3" 或 "10 / 2"
        required: true
"""

import math
import operator
import ast


def run(expression: str):
    """
    安全地计算数学表达式
    仅允许基本运算节点
    """
    try:
        # 使用 ast 解析并评估表达式，限制为安全操作
        node = ast.parse(expression, mode='eval')

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.Pow):
                    return left ** right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                else:
                    raise ValueError(f"不支持的操作: {type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                elif isinstance(node.op, ast.UAdd):
                    return +operand
                else:
                    raise ValueError(f"不支持的一元操作: {type(node.op).__name__}")
            elif isinstance(node, ast.Call):
                # 允许调用部分安全函数
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                allowed_funcs = {
                    'abs': abs,
                    'round': round,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'pow': pow,
                    'sqrt': math.sqrt,
                    'sin': math.sin,
                    'cos': math.cos,
                    'tan': math.tan,
                    'log': math.log,
                    'log10': math.log10,
                    'exp': math.exp,
                    'ceil': math.ceil,
                    'floor': math.floor,
                    'pi': math.pi,
                    'e': math.e,
                }
                if func_name not in allowed_funcs:
                    raise ValueError(f"不允许调用的函数: {func_name}")
                args = [_eval(arg) for arg in node.args]
                return allowed_funcs[func_name](*args)
            elif isinstance(node, ast.Name):
                allowed_names = {
                    'pi': math.pi,
                    'e': math.e,
                    'inf': math.inf,
                    'nan': math.nan,
                }
                if node.id not in allowed_names:
                    raise ValueError(f"不允许使用的名称: {node.id}")
                return allowed_names[node.id]
            else:
                raise ValueError(f"不支持的节点类型: {type(node).__name__}")

        result = _eval(node)
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except ZeroDivisionError:
        return {"error": "除零错误"}
    except Exception as e:
        return {"error": str(e)}
