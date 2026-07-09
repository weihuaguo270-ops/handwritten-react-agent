"""
Python 代码执行 — 让 Agent 写代码并运行

让 Agent 从"查询型"变成"执行型"：
  用户: "分析数据并画图"
  Agent 写 Python 脚本 → 执行 → 看输出 → 迭代 → 最终结果

安全机制：
  - 子进程隔离执行（不阻塞主 Agent）
  - 硬超时限制（默认 30s，防止死循环）
  - 只暴露 stdout/stderr，不共享文件系统之外的资源
"""
import subprocess
import sys
import os
import tempfile
import json
import time


def execute_python(code: str, timeout: int = 30) -> str:
    """
    执行 Python 代码并返回输出。

    参数:
        code: 要执行的 Python 代码（字符串）
        timeout: 超时秒数（默认 30，最大 120）

    返回:
        stdout + stderr（或超时/错误信息）

    可用包: numpy, scikit-learn, sentence-transformers, flask, requests, matplotlib
    如需安装新包，请先尝试 pip install，若失败则提示用户手动安装。
    """
    # 安全限制
    timeout = min(timeout, 120)

    # 写入临时文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        start = time.time()
        r = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # 限制工作目录为临时目录，防止误写项目文件
            cwd=os.path.dirname(tmp_path),
        )
        elapsed = time.time() - start

        output_parts = []
        if r.stdout.strip():
            output_parts.append(r.stdout.strip())
        if r.stderr.strip():
            output_parts.append(f"[stderr]\n{r.stderr.strip()}")

        if not output_parts:
            output_parts.append("（代码执行完毕，无输出）")

        output_parts.append(f"\n[状态] 退出码 {r.returncode}，耗时 {elapsed:.2f}s")
        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"[错误] 代码执行超时（{timeout}s），请检查是否有死循环或优化算法"
    except Exception as e:
        return f"[错误] 执行失败: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "执行 Python 代码并返回 stdout/stderr。"
                       "可以写数据分析、文件处理、图表生成、Web 请求等任意 Python 脚本。"
                       "代码会在隔离子进程中运行，有超时保护。"
                       "可用库：numpy, scikit-learn, sentence-transformers, flask, requests, matplotlib。"
                       "如需其他库，先 pip install 再重试。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "完整的 Python 代码。"
                                   "用 print() 输出结果，不支持 input() 交互输入。"
                                   "注意：工作目录是临时目录，读写文件请用完整路径或 print 到 stdout。",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数，默认 30，最大 120",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    },
}
