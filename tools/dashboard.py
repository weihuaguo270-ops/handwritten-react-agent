"""启动 Dashboard Web 界面（轨迹查看器 + 聊天面板）"""
import os
import subprocess
import sys


DASHBOARD_PROCESS = []


def start_dashboard(port: int = 5050) -> str:
    """启动 Dashboard Web 界面（轨迹查看器 + 聊天面板）"""
    # 先杀掉旧进程
    try:
        kill_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "dashboard", "kill_old.py")
        if os.path.exists(kill_script):
            subprocess.run([sys.executable, kill_script],
                          cwd=os.path.dirname(kill_script),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        pass
    try:
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "dashboard", "server.py")
        if not os.path.exists(server_script):
            return f"错误: 找不到 dashboard/server.py"
        proc = subprocess.Popen(
            [sys.executable, server_script],
            cwd=os.path.dirname(server_script),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        DASHBOARD_PROCESS.append(proc)
        return f"Dashboard 已启动: http://127.0.0.1:{port}（记得在浏览器打开）"
    except Exception as e:
        return f"启动失败: {e}"


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "start_dashboard",
        "description": "启动 Dashboard Web 界面（轨迹查看器 + 聊天面板），在浏览器中直观查看 Agent 的思考过程和工具调用",
        "parameters": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "端口号（默认 5050）"
                }
            },
            "required": []
        }
    },
}
