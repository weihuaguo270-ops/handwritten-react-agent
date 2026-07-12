"""计算数学表达式（基于 AST 解析，安全无注入风险）"""
import ast


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        tree = ast.parse(expression.strip(), mode='eval')
        ALLOWED_NODES = (ast.Expression, ast.Constant, ast.UnaryOp,
                         ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div)
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return "错误：表达式包含非法操作"
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        return str(result)
    except SyntaxError:
        return "错误：表达式语法错误"
    except Exception as e:
        return f"计算错误：{e}"


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        },
    },
}
