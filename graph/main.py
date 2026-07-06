"""
CLI 入口 — 替代手写 react_loop.py 中的 main()

交互模式 + 单次查询模式 + MCP 连接
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # graph/

from agent import run as run_agent
from memory import MEMORY
from rag import ingest_directory


def main():
    """主入口：支持命令行查询和交互模式"""
    # 启动时索引项目文档到 RAG
    _rag_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("[启动] 正在加载 RAG 知识库...")
    try:
        ingest_directory(_rag_dir)
    except Exception as e:
        print(f"[启动] RAG 加载跳过: {e}")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        _handle_single_query(q)
    else:
        _interactive_mode()


def _handle_single_query(q: str):
    """单次查询模式"""
    if "忘记" in q or "删除" in q:
        target = q.split("忘记", 1)[1].strip() if "忘记" in q else q.split("删除", 1)[1].strip()
        if "所有" in target or "全部" in target:
            MEMORY.clear()
            print("\n[记忆] 已清空所有记忆")
        elif target:
            MEMORY.remove(target)
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
    result = run_agent(full_q)
    if result:
        print(f"\n最终答案: {result}")


def _interactive_mode():
    """交互式 CLI 模式"""
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
