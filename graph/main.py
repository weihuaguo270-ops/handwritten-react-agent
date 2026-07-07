"""
CLI 入口 — 交互模式 + 单次查询模式 + MCP 连接 + Harness 轨迹记录

MCP 连接：启动时自动连接 mcp-server-time（时间查询工具）。
Harness：每次对话自动记录完整轨迹（thought → action → observation），
         保存 JSON 到 trajectories/ 目录，支持 Replay 回放。
"""

import sys
import os
import json
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from mcp import connect_default_servers
from agent import run as run_agent
from memory import MEMORY
from rag import ingest_directory

# Harness 集成
from harness import Harness

# MCP 全局客户端列表（供 agent.py 的 tools_node 转发 MCP 调用）
MCP_CLIENTS = []

# 轨迹目录（与手写版共享）
TRAJECTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trajectories",
)


# ============================================================
# MCP 连接
# ============================================================

def _connect_mcp():
    """启动时连接 MCP 服务器"""
    global MCP_CLIENTS
    MCP_CLIENTS = connect_default_servers()


# ============================================================
# Harness 集成
# ============================================================

def _get_harness(enable_sandbox: bool = False) -> Harness:
    """创建配置好的 Harness 实例

    graph/ 版本的沙箱默认关闭（开发阶段），需通过环境变量 GRAPH_ENABLE_SANDBOX=true 开启。
    快速工具（get_current_time, calculator）注册为 unsafe，不通过子进程执行。
    """
    sandbox_env = os.environ.get("GRAPH_ENABLE_SANDBOX", "false").lower()
    enable_sandbox = enable_sandbox or sandbox_env in ("true", "1", "yes")

    harness = Harness(
        sandbox_enabled=False,  # 默认关闭——Agent 循环中的工具执行不通过子进程
        sandbox_timeout=30,
    )
    # 注册快速工具，不走沙箱
    harness.add_unsafe_tool("get_current_time")
    harness.add_unsafe_tool("calculator")

    return harness


# ============================================================
# 主入口
# ============================================================

def main():
    _rag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("[启动] 正在加载 RAG 知识库...")
    try:
        ingest_directory(_rag_dir)
    except Exception as e:
        print(f"[启动] RAG 加载跳过: {e}")

    print("[启动] 正在连接 MCP 服务器...")
    _connect_mcp()

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        _handle_single_query(q)
    else:
        _interactive_mode()


def _handle_single_query(q: str):
    """处理单次查询，自动创建 Harness 记录轨迹"""
    if _handle_memory_ops(q):
        return

    # ── 创建 Harness 记录本次对话轨迹 ──
    harness = _get_harness()
    harness.start_trajectory(query=q, model="deepseek-chat",
                             system_prompt="LangChain Agent 交互模式")

    memories = MEMORY.query(q)
    memory_context = ""
    if memories:
        memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])

    full_q = memory_context + q if memory_context else q
    result = run_agent(full_q, mcp_clients=MCP_CLIENTS, harness=harness)

    if result:
        print(f"\n最终答案: {result}")

    # ── 保存轨迹 ──
    harness.finish(final_answer=result or "")
    saved_path = harness.save()
    if saved_path:
        steps = len(harness.recorder.steps) if harness.recorder else 0
        print(f"[Harness] 轨迹已保存: {os.path.basename(saved_path)} （{steps} 步）")


def _interactive_mode():
    """交互模式，每轮对话自动创建 Harness"""
    print("\n" + "=" * 50)
    print("  LangChain Agent 交互模式")
    print("=" * 50)
    print("  退出：输入 'exit' 或 '退出'")
    print("  查看记忆：输入 '记忆'")
    print("  Harness 轨迹自动记录到 trajectories/ 目录")
    print("=" * 50)

    session_count = 0

    while True:
        q = input("\n你 > ").strip()
        if q.lower() in ("exit", "退出", "quit"):
            # 会话结束时显示汇总
            print(f"再见！本次共 {session_count} 轮对话，轨迹文件在 {TRAJECTORY_DIR}")
            break
        if not q:
            continue
        if q == "记忆":
            print("\n已保存的记忆:")
            if MEMORY.facts:
                for i, f in enumerate(MEMORY.facts, 1):
                    print(f"  {i}. {f}")
            else:
                print("  （无）")
            continue

        # 轨迹记录 + 沙箱
        harness = _get_harness()
        harness.start_trajectory(query=q, model="deepseek-chat",
                                 system_prompt="LangChain Agent 交互模式")

        memories = MEMORY.query(q)
        memory_context = ""
        if memories:
            memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])

        full_q = memory_context + q if memory_context else q
        result = run_agent(full_q, mcp_clients=MCP_CLIENTS, harness=harness)

        if result:
            print(f"\n最终答案: {result}")

        harness.finish(final_answer=result or "")
        saved_path = harness.save()
        if saved_path:
            steps = len(harness.recorder.steps) if harness.recorder else 0
            print(f"[Harness] 轨迹已保存: {os.path.basename(saved_path)} （{steps} 步）")

        session_count += 1


# ============================================================
# 记忆操作快捷函数
# ============================================================

def _handle_memory_ops(q: str) -> bool:
    """处理记忆相关的快捷命令（忘记/记住），返回 True 表示已处理"""
    if "忘记" in q or "删除" in q:
        target = q.split("忘记", 1)[1].strip() if "忘记" in q else q.split("删除", 1)[1].strip()
        if target in ("所有", "全部"):
            MEMORY.clear()
            print("\n[记忆] 已清空所有记忆")
        elif target:
            MEMORY.remove(target)
        return True

    if "记住" in q:
        fact = q.split("记住", 1)[1].strip().lstrip(" ，,、。.：:")
        if fact:
            action, detail = MEMORY.add_or_update(fact)
            if action == "skipped":
                print(f"\n[记忆] 已存在: {fact}")
            elif action == "updated":
                print(f"\n[记忆] 已更新: \"{detail}\" → \"{fact}\"")
            else:
                print(f"\n[记忆] 已记住: {fact}")
        return True

    return False


if __name__ == "__main__":
    main()
