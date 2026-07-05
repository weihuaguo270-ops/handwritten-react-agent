"""获取当前时间"""
import time


def get_time() -> str:
    """返回当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "获取当前日期和时间",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
