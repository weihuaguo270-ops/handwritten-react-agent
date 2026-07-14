"""
Python 代码执行 — 让 Agent 写代码并运行

让 Agent 从"查询型"变成"执行型"：
  用户: "分析数据并画图"
  Agent 写 Python 脚本 → 执行 → 看输出 → 迭代 → 最终结果

隔离说明（学习用，非安全沙箱）：
  - 在独立子进程中运行，带硬超时（默认 30s），避免拖死主 Agent
  - 只回传 stdout/stderr
  - cwd 设为临时目录，降低误改仓库文件的概率
  - **不能**阻止恶意代码读写本机可达文件、访问网络或耗尽资源
  - 请勿对不可信第三方代码开启本工具
"""
import subprocess
import sys
import os
import tempfile
import time


def execute_python(code: str, timeout: int = 30) -> str:
    """
    执行 Python 代码并返回输出。

    参数:
        code: 要执行的 Python 代码（字符串）
        timeout: 超时秒数（默认 30，最大 120）

    返回:
        stdout + stderr（或超时/错误信息）

    可用包取决于当前解释器环境；语义检索相关包见 `pip install -e ".[rag]"`。
    """
    # 超时上限（软限制，非资源配额）
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
            # 工作目录为临时目录，降低误写项目文件的概率（仍非安全边界）
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
        "description": (
            "在子进程中执行 Python 代码并返回 stdout/stderr。"
            "仅用于信任环境下的计算/脚本；不是安全沙箱。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 源代码",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数，默认 30，最大 120",
                },
            },
            "required": ["code"],
        },
    },
}
