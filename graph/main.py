"""
CLI 入口 — 交互模式 + 单次查询模式 + MCP 连接 + 轨迹记录

MCP 连接：启动时自动连接 mcp-server-time（时间查询工具）。
轨迹记录：每次对话自动保存轨迹 JSON 到 trajectories/ 目录。
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

# MCP 全局客户端列表（供 agent.py 的 tools_node 转发 MCP 调用）
MCP_CLIENTS = []

# 轨迹目录
TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trajectories")


# ============================================================
# MCP 连接
# ============================================================

def _connect_mcp():
    """启动时连接 MCP 服务器"""
    global MCP_CLIENTS
    MCP_CLIENTS = connect_default_servers()


# ============================================================
# 轨迹记录
# ============================================================

def _save_trajectory(query: str, result: str, messages_count: int):
    """保存对话轨迹到 JSON 文件"""
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_graph.json"
    path = os.path.join(TRAJECTORY_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "result": result,
            "messages_count": messages_count,
            "source": "graph",
        }, f, ensure_ascii=False, indent=2)
    # 清理旧轨迹（保留最近 100 条）
    all_files = sorted(glob.glob(os.path.join(TRAJECTORY_DIR, "*_graph.json")))
    while len(all_files) > 100:
        os.remove(all_files.pop(0))


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
    if "忘记" in q or "删除" in q:
        target = q.split("忘记", 1)[1].strip() if "忘记" in q else q.split("删除", 1)[1].strip()
        if target in ("所有", "全部"):
            MEMORY.clear()
            print("\n[记忆] 已清空所有记忆")
        elif target:
            MEMORY.remove(target)
        _save_trajectory(q, f"[记忆操作] {'清空' if target in ('所有','全部') else '删除:' + target}", 0)
        return

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
        return

    memories = MEMORY.query(q)
    memory_context = ""
    if memories:
        memory_context = "\n".join([f"（相关记忆：{m['fact']}）" for m in memories])

    full_q = memory_context + q if memory_context else q
    result = run_agent(full_q, mcp_clients=MCP_CLIENTS)
    if result:
        print(f"\n最终答案: {result}")
        _save_trajectory(q, result, 0)


def _interactive_mode():
    print("\n" + "=" * 50)
    print("  LangChain Agent 交互模式")
    print("=" * 50)
    print("  退出：输入 'exit' 或 '退出'")
    print("  查看记忆：输入 '记忆'")
    print("=" * 50)

    while True:
        q = input("\n你 > ").strip()
        if q.lower() in ("exit", "退出", "quit"):
            print("再见！")
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

        _handle_single_query(q)


if __name__ == "__main__":
    main()
