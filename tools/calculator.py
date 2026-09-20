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

# 安全限制：防止 DoS 攻击（如 2**2**100 会导致内存耗尽）
_MAX_INTERMEDIATE = 10**15  # 中间结果最大绝对值
_MAX_DEPTH = 100  # AST 递归深度限制


def run(expression: str):
    """
    安全地计算数学表达式
    仅允许基本运算节点，并对中间结果大小和递归深度做限制，
    防止恶意表达式（如 2**2**100）导致 DoS。
    """
    try:
        # 限制表达式长度，防止超长输入
        if len(expression) > 1000:
            return {"error": "表达式过长（超过1000字符），请简化后重试"}

        # 使用 ast 解析并评估表达式，限制为安全操作
        node = ast.parse(expression, mode='eval')

        def _eval(node, depth=0):
            if depth > _MAX_DEPTH:
                raise ValueError("表达式嵌套过深，可能存在 DoS 风险")

            if isinstance(node, ast.Expression):
                return _eval(node.body, depth)
            elif isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, (int, float)) and abs(value) > _MAX_INTERMEDIATE:
                    raise ValueError(f"数值 {value} 超出安全范围（±{_MAX_INTERMEDIATE}）")
                return value
            elif isinstance(node, ast.Num):  # Python < 3.8
                value = node.n
                if abs(value) > _MAX_INTERMEDIATE:
                    raise ValueError(f"数值 {value} 超出安全范围（±{_MAX_INTERMEDIATE}）")
                return value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left, depth + 1)
                right = _eval(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.Pow):
                    # Pow 操作特殊限制：底数和指数都不能太大
                    if isinstance(right, (int, float)) and abs(right) > 1000:
                        raise ValueError(f"指数 {right} 超出安全范围（±1000），可能导致 DoS")
                    if isinstance(left, (int, float)) and abs(left) > _MAX_INTERMEDIATE:
                        raise ValueError(f"底数 {left} 超出安全范围（±{_MAX_INTERMEDIATE}）")
                    result = left ** right
                elif isinstance(node.op, ast.Mod):
                    result = left % right
                elif isinstance(node.op, ast.FloorDiv):
                    result = left // right
                else:
                    raise ValueError(f"不支持的操作: {type(node.op).__name__}")

                # 检查中间结果是否超出安全范围
                if isinstance(result, (int, float)) and not math.isnan(result) and not math.isinf(result):
                    if abs(result) > _MAX_INTERMEDIATE:
                        raise ValueError(f"计算中间结果 {result} 超出安全范围（±{_MAX_INTERMEDIATE}）")
                return result
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand, depth + 1)
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
                args = [_eval(arg, depth + 1) for arg in node.args]
                # pow 函数也需要安全限制
                if func_name == 'pow' and len(args) >= 2:
                    if abs(args[1]) > 1000:
                        raise ValueError(f"pow 指数 {args[1]} 超出安全范围（±1000）")
                result = allowed_funcs[func_name](*args)
                if isinstance(result, (int, float)) and not math.isnan(result) and not math.isinf(result):
                    if abs(result) > _MAX_INTERMEDIATE:
                        raise ValueError(f"函数结果 {result} 超出安全范围（±{_MAX_INTERMEDIATE}）")
                return result
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
    except OverflowError:
        return {"error": "数值溢出，表达式结果过大"}
    except Exception as e:
        return {"error": str(e)}
