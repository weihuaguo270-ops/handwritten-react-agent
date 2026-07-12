"""
Tree of Thoughts (ToT) — 思维树多路径推理

从手写版 src/react_agent/tot.py 迁移，适配 LangGraph 的 LLM 调用方式。
"""
from __future__ import annotations
from typing import Optional, Callable
import json


# ── 搜索策略 ──

class SearchStrategy:
    BFS = "bfs"
    DFS = "dfs"


# ── 树节点 ──

class ToTNode:
    def __init__(self, content: str, parent: Optional["ToTNode"] = None):
        self.content = content
        self.score: Optional[float] = None
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.children: list[ToTNode] = []
        if parent:
            parent.children.append(self)

    def path_to_root(self) -> list[str]:
        """从根节点到当前节点的路径"""
        path = []
        node = self
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))

    def best_leaf(self) -> "ToTNode":
        """递归找到评分最高的叶子节点"""
        if not self.children:
            return self
        best = max((child.best_leaf() for child in self.children),
                   key=lambda n: n.score or 0)
        return best

    def __repr__(self):
        return f"ToTNode(depth={self.depth}, score={self.score}, content={self.content[:40]}...)"


# ── Prompt 模板 ──

GENERATE_PROMPT = """你正在解决一个需要多步骤推理的问题。

当前进度：
{path}

请生成下一步推理，只输出推理内容，不要评分。"""

EVALUATE_PROMPT = """评估以下推理步骤的质量（0-10 分）。

问题：{query}
推理步骤：{step}

评分标准：
- 10：完全正确且合理的推理
- 7-9：大体正确，但有些遗漏
- 4-6：部分合理，但存在逻辑问题
- 1-3：基本不合理
- 0：完全错误

只输出数字评分（0-10）："""


# ── ToT 求解器 ──

class TreeOfThoughts:
    """思维树求解器

    用法：
        tot = TreeOfThoughts(llm_call=lambda prompt: call_llm(prompt))
        result = tot.solve("复杂的数学问题", strategy="bfs")
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        beam_width: int = 3,
        max_depth: int = 3,
    ):
        self.llm_call = llm_call
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.root = ToTNode("问题：" + "待求解")
        self._calls = 0

    @property
    def llm_calls(self) -> int:
        return self._calls

    def solve(self, query: str, strategy: str = "bfs") -> str:
        """执行 ToT 推理

        参数:
            query: 待解决的问题
            strategy: "bfs"（广度优先）或 "dfs"（深度优先）
        """
        self.root = ToTNode(query)
        self._calls = 0

        if strategy == SearchStrategy.DFS:
            self._dfs(self.root)
        else:
            self._bfs()

        best = self.root.best_leaf()
        return "\n".join(best.path_to_root())

    def _generate_thoughts(self, node: ToTNode) -> list[str]:
        """从当前节点生成多个下一步推理"""
        path_text = "\n".join(node.path_to_root())
        prompt = GENERATE_PROMPT.format(path=path_text)

        try:
            raw = self.llm_call(prompt)
            self._calls += 1
        except Exception as e:
            return [f"（生成失败: {e}）"]

        thoughts = [t.strip() for t in raw.split("\n") if t.strip()
                    and not t.startswith("```")]
        return thoughts[:self.beam_width] if thoughts else [raw[:200]]

    def _evaluate(self, node: ToTNode, query: str) -> float:
        """评估节点的质量"""
        prompt = EVALUATE_PROMPT.format(query=query, step=node.content)
        try:
            raw = self.llm_call(prompt)
            self._calls += 1
            score = float("".join(c for c in raw if c.isdigit() or c == ".")[:4])
            return max(0, min(10, score))
        except (ValueError, IndexError):
            return 5.0

    def _bfs(self):
        """广度优先搜索"""
        current_level = [self.root]

        for depth in range(self.max_depth):
            next_level = []

            for node in current_level:
                thoughts = self._generate_thoughts(node)
                for t in thoughts:
                    child = ToTNode(t, parent=node)
                    child.score = self._evaluate(child, self.root.content)
                    next_level.append(child)

            # 按分数保留 top-K
            next_level.sort(key=lambda n: n.score or 0, reverse=True)
            current_level = next_level[:self.beam_width]

            if not current_level:
                break

    def _dfs(self, node: ToTNode, depth: int = 0):
        """深度优先搜索"""
        if depth >= self.max_depth:
            node.score = self._evaluate(node, self.root.content)
            return

        thoughts = self._generate_thoughts(node)
        # 按评分排序探索
        scored = []
        for t in thoughts:
            child = ToTNode(t, parent=node)
            child.score = self._evaluate(child, self.root.content)
            scored.append(child)

        scored.sort(key=lambda n: n.score or 0, reverse=True)

        for child in scored[:self.beam_width]:
            self._dfs(child, depth + 1)

    def stats(self) -> dict:
        return {
            "llm_calls": self._calls,
            "nodes": self._count_nodes(self.root),
            "depth": self._max_depth(self.root),
        }

    @staticmethod
    def _count_nodes(node: ToTNode) -> int:
        return 1 + sum(TreeOfThoughts._count_nodes(c) for c in node.children)

    @staticmethod
    def _max_depth(node: ToTNode) -> int:
        if not node.children:
            return node.depth
        return max(TreeOfThoughts._max_depth(c) for c in node.children)
