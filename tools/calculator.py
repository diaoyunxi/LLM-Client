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
                # 防止超大数值导致 CPU 耗尽（DoS）
                _MAX_OPERAND = 10**15
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    if abs(left) > _MAX_OPERAND or abs(right) > _MAX_OPERAND:
                        raise ValueError(f"操作数过大（上限 {_MAX_OPERAND}），拒绝计算")
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ZeroDivisionError("除数不能为零")
                    return left / right
                elif isinstance(node.op, ast.Pow):
                    # 幂运算额外限制：底数和指数都不能过大
                    if isinstance(left, (int, float)) and abs(left) > 1000:
                        raise ValueError("幂运算底数不能超过 1000")
                    if isinstance(right, (int, float)) and abs(right) > 1000:
                        raise ValueError("幂运算指数不能超过 1000")
                    return left ** right
                elif isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ZeroDivisionError("取模除数不能为零")
                    return left % right
                elif isinstance(node.op, ast.FloorDiv):
                    if right == 0:
                        raise ZeroDivisionError("整除除数不能为零")
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
                def _safe_pow(*args):
                    if len(args) >= 2 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                        if abs(args[0]) > 1000 or abs(args[1]) > 1000:
                            raise ValueError("pow() 参数不能超过 1000")
                    return pow(*args)

                allowed_funcs = {
                    'abs': abs,
                    'round': round,
                    'max': max,
                    'min': min,
                    'sum': sum,
                    'pow': _safe_pow,
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
