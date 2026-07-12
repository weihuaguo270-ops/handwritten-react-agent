"""
思维树（Tree-of-Thought）推理模块

比 CoT 更进一步——同时探索多条推理路径，评估打分，剪枝回溯。
适用于需要多方案比较的复杂问题（逻辑题、规划题、数学题）。

和 ReAct Loop 的关系:
  ToT 是 ReAct "思考"阶段的增强模块。
  当 LLM 判断问题需要多路径探索时，调用 tot_reasoning 工具。
  tot.solve() 内部多次调 LLM 做生成+评估，返回最终答案。

搜索策略:
  BFS (广度优先): 每层保留 beam_width 条路径，适合搜索空间宽的题
  DFS (深度优先): 一条路到底，不行再回溯，适合步骤深但分支少的题
"""

from enum import Enum
from typing import Optional, Callable
import re


# ============================================================
# 1. 搜索策略枚举
# ============================================================
class SearchStrategy(Enum):
    BFS = "bfs"   # 广度优先
    DFS = "dfs"   # 深度优先


# ============================================================
# 2. 树节点
# ============================================================
class ToTNode:
    """思维树中的一个节点——代表一步推理

    属性:
        content: 这一步的推理内容
        score:   评分 (0-10)，None 表示还没评
        parent:  父节点（None 表示根节点）
        depth:   深度（根节点 = 0）
        children: 子节点列表
    """
    def __init__(self, content: str, parent: Optional["ToTNode"] = None):
        self.content = content
        self.score = None
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def chain(self) -> list[str]:
        """从根节点到当前节点的完整推理链"""
        if self.parent is None:
            return [self.content] if self.content else []
        return self.parent.chain() + [self.content]

    def best_child(self) -> Optional["ToTNode"]:
        """返回评分最高的子节点"""
        scored = [c for c in self.children if c.score is not None]
        return max(scored, key=lambda c: c.score) if scored else None

    def __repr__(self):
        score_str = f" ({self.score}/10)" if self.score is not None else ""
        return f"<ToTNode depth={self.depth} score={self.score}{score_str}>"


# ============================================================
# 3. ToT 核心类
# ============================================================

_TOO_MANY_CALLS_WARN = 50  # 超过这个次数报警

class ToT:
    """思维树推理引擎

    用法:
        tot = ToT(beam_width=3, branch_factor=3, max_depth=5)

        def my_llm(prompt: str) -> str:
            ...  # 你的 LLM 调用逻辑

        answer = tot.solve("用 4 步把 1 2 3 4 运算得到 24", llm_call=my_llm)
    """

    def __init__(self, beam_width: int = 3, branch_factor: int = 3,
                 max_depth: int = 5, strategy: SearchStrategy = SearchStrategy.BFS):
        self.beam_width = beam_width        # 每层保留多少条路径
        self.branch_factor = branch_factor  # 每个节点生成多少个候选
        self.max_depth = max_depth          # 最大搜索深度
        self.strategy = strategy            # BFS / DFS
        self._llm_call_count = 0            # 统计 LLM 调用次数

    # ----------------------------------------------------------
    # 3a. LLM 调用的 prompt 模板
    # ----------------------------------------------------------

    _GENERATE_PROMPT = """你正在解决一个需要逐步推理的问题。

问题: {problem}

当前已经完成的推理步骤：
{chain_text}

请思考基于当前已知信息，下一步应该推理什么。
注意：
- 每一步应该利用上一步得出的信息来推进推理，不要原地重复
- 如果你已经收集了足够的信息来得出结论，输出步骤以 "ANSWER:" 开头给出最终答案
- 对每张卡片/每个条件逐一分析，不要只分析一个就停下

生成 {num_candidates} 种不同的可能下一步，每步1-2句话。
每步用 --- 分隔。

输出格式：
步骤1: ...
---
步骤2: ...
---
步骤3: ...

直接输出结果，不要额外解释。"""

    _EVALUATE_PROMPT = """你是一个严谨的推理评分员。

问题: {problem}

当前推理链：
{chain_text}

候选下一步：{candidate}

请评估这个候选步骤对解决问题的价值，从 0 到 10 分：
- 0-3: 没有帮助，方向错误或无关（例如重复已经做过的步骤、原地打转）
- 4-6: 有一定价值，但不够关键（只是列举动作，没有推理分析）
- 7-10: 非常有价值，是解决问题的关键步骤（推进了推理、分析了条件、得出结论）

关键评分标准：
- 如果这个步骤是在"分析条件、推理结论"而不是"简单地列举下一张要翻的卡"，给高分
- 如果这步是在原地重复同样类型的动作（比如继续翻另一张卡而不分析），给低分
- 如果这步输出了最终结论（ANSWER:），且结论合理，给 9-10 分

只输出一个整数评分（0-10）："""

    # ----------------------------------------------------------
    # 3b. 核心方法
    # ----------------------------------------------------------

    def solve(self, problem: str, llm_call: Optional[Callable] = None) -> dict:
        """主入口：运行 ToT 推理

        参数:
            problem: 要解决的问题
            llm_call: 接收 prompt 字符串、返回 LLM 回复文本的函数

        返回:
            {"answer": str, "chain": list[str], "llm_calls": int,
             "scores": list[int], "error": str | None}
        """
        if llm_call is None:
            return {"answer": "", "chain": [], "llm_calls": 0,
                    "scores": [], "error": "未提供 llm_call 函数"}

        self._llm_call_count = 0

        try:
            root = ToTNode("")  # 根节点（空内容）
            if self.strategy == SearchStrategy.BFS:
                result = self._search_bfs(problem, root, llm_call)
            else:
                result = self._search_dfs(problem, root, llm_call, 0)

            result["llm_calls"] = self._llm_call_count
            if not result.get("error"):
                result.pop("error", None)
            return result

        except Exception as e:
            return {
                "answer": "",
                "chain": [],
                "llm_calls": self._llm_call_count,
                "scores": [],
                "error": str(e),
            }

    def reset(self):
        """重置 LLM 调用计数"""
        self._llm_call_count = 0

    # ----------------------------------------------------------
    # 3c. BFS 搜索
    # ----------------------------------------------------------

    def _search_bfs(self, problem: str, root: ToTNode,
                    llm_call: Callable) -> dict:
        """广度优先搜索——每层保留 top-K 路径"""
        current_level = [root]
        best_answer = ""
        best_chain = []
        best_score = -1

        for depth in range(1, self.max_depth + 1):
            if self._llm_call_count > _TOO_MANY_CALLS_WARN:
                break

            # 当前层所有节点的所有候选
            all_candidates: list[tuple[ToTNode, str]] = []

            for node in current_level:
                chain_text = self._format_chain(node.chain())
                prompt = self._GENERATE_PROMPT.format(
                    problem=problem,
                    chain_text=chain_text,
                    num_candidates=self.branch_factor,
                )
                reply = llm_call(prompt)
                self._llm_call_count += 1

                candidates = self._parse_candidates(reply)
                for c in candidates:
                    all_candidates.append((node, c))

            if not all_candidates:
                break

            # 评估每个候选
            scored: list[tuple[float, ToTNode]] = []
            for parent_node, candidate in all_candidates:
                chain_text = self._format_chain(parent_node.chain())
                eval_prompt = self._EVALUATE_PROMPT.format(
                    problem=problem,
                    chain_text=chain_text,
                    candidate=candidate,
                )
                score_reply = llm_call(eval_prompt)
                self._llm_call_count += 1
                score = self._parse_score(score_reply)

                child = ToTNode(candidate, parent=parent_node)
                child.score = score
                scored.append((score, child))

                # 检查是否可能是最终答案（包含最终答案标记）
                if ("FINAL ANSWER" in candidate.upper() or candidate.upper().startswith("ANSWER:")) and score >= 7:
                    return {
                        "answer": candidate,
                        "chain": child.chain(),
                        "scores": [score],
                        "error": None,
                    }

            # 按评分排序，保留 top-K
            scored.sort(key=lambda x: x[0], reverse=True)
            current_level = [node for _, node in scored[:self.beam_width]]

            # 记录当前最优
            if scored and scored[0][0] > best_score:
                best_score = scored[0][0]
                best_chain = scored[0][1].chain()
                best_answer = scored[0][1].content

            # 如果最优路径评分足够高且深度够，提前结束
            if best_score >= 9 and depth >= 2:
                break

        return {
            "answer": best_answer,
            "chain": best_chain,
            "scores": [best_score] if best_score >= 0 else [],
            "error": None,
        }

    # ----------------------------------------------------------
    # 3d. DFS 搜索
    # ----------------------------------------------------------

    def _search_dfs(self, problem: str, node: ToTNode,
                    llm_call: Callable, depth: int) -> dict:
        """深度优先搜索——递归，不行就回溯"""
        if depth >= self.max_depth:
            return {"answer": node.content if node.parent else "",
                    "chain": node.chain(), "scores": [],
                    "error": None}

        if self._llm_call_count > _TOO_MANY_CALLS_WARN:
            return {"answer": node.content, "chain": node.chain(),
                    "scores": [], "error": "超过 LLM 调用上限"}

        chain_text = self._format_chain(node.chain())
        prompt = self._GENERATE_PROMPT.format(
            problem=problem,
            chain_text=chain_text,
            num_candidates=self.branch_factor,
        )
        reply = llm_call(prompt)
        self._llm_call_count += 1
        candidates = self._parse_candidates(reply)

        # 评估并排序
        scored = []
        for c in candidates:
            eval_prompt = self._EVALUATE_PROMPT.format(
                problem=problem,
                chain_text=chain_text,
                candidate=c,
            )
            score_reply = llm_call(eval_prompt)
            self._llm_call_count += 1
            score = self._parse_score(score_reply)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 从高分到低分逐个探索
        best_result = {"answer": "", "chain": [], "scores": [], "error": None}
        best_score = -1

        for score, candidate in scored[:self.beam_width]:
            child = ToTNode(candidate, parent=node)
            child.score = score

            if candidate.upper().startswith("FINAL ANSWER"):
                return {"answer": candidate, "chain": child.chain(),
                        "scores": [score], "error": None}

            result = self._search_dfs(problem, child, llm_call, depth + 1)
            if result.get("answer") and result.get("scores"):
                avg_score = sum(result["scores"]) / len(result["scores"])
                if avg_score > best_score:
                    best_result = result
                    best_score = avg_score

        return best_result if best_result["answer"] else {
            "answer": scored[0][1] if scored else "",
            "chain": node.chain() + ([scored[0][1]] if scored else []),
            "scores": [scored[0][0]] if scored else [],
            "error": None,
        }

    # ----------------------------------------------------------
    # 3e. 辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _format_chain(chain: list[str]) -> str:
        """把推理链格式化成可读的文本"""
        if not chain:
            return "（还没有进行任何推理步骤）"
        parts = []
        for i, step in enumerate(chain, 1):
            if step.strip():
                parts.append(f"第{i}步: {step.strip()}")
        return "\n".join(parts) if parts else "（还没有进行任何推理步骤）"

    @staticmethod
    def _parse_candidates(reply: str) -> list[str]:
        """从 LLM 回复中解析出候选步骤列表"""
        candidates = []

        # 方法1：按 --- 分隔
        parts = re.split(r"\n---+\n", reply)
        for part in parts:
            part = part.strip()
            # 去掉可能的前缀 "步骤N:"
            part = re.sub(r"^步骤\d+[:：]\s*", "", part)
            if part:
                candidates.append(part)

        # 如果解析不到内容，整段作为一个候选
        if not candidates:
            cleaned = reply.strip()
            if cleaned:
                candidates = [cleaned]

        return candidates

    @staticmethod
    def _parse_score(reply: str) -> float:
        """从 LLM 回复中解析出评分 (0-10)"""
        # 方法1：直接找数字
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", reply)
        for n in nums:
            val = float(n)
            if 0 <= val <= 10:
                return val

        # 方法2：找 "N/10" 模式
        frac = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", reply)
        if frac:
            return float(frac.group(1))

        return 5.0  # 无法解析时给个中等分


# ============================================================
# 4. 全局实例
# ============================================================

TOT = ToT()


# ============================================================
# 5. 工具定义（供 react_loop.py 注册）
# ============================================================

TOT_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tot_reasoning",
        "description": "对复杂问题使用思维树（Tree-of-Thought）进行多路径推理。"
                       "适合需要多方案比较、分步探索的问题，如逻辑推理、规划、数学难题。"
                       "内部会多次调 LLM，对每个候选路径进行评估打分。",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "需要解决的问题描述，越详细越好"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "搜索深度（默认5），复杂问题可以设大一些",
                    "default": 5,
                },
            },
            "required": ["problem"],
        },
    },
}


def tool_tot_reasoning(problem: str, max_depth: int = 5) -> str:
    """运行时调用 ToT 推理——会被注册为工具供 LLM 调用

    注意: 这个工具需要从 react_loop.py 中注入 call_llm 函数。
    通过 set_tot_llm_call() 在初始化时设置。
    """
    global _tot_llm_call
    if _tot_llm_call is None:
        return "错误: ToT 的 LLM 调用函数未设置，请先调用 set_tot_llm_call()"

    tot = ToT(beam_width=3, branch_factor=3, max_depth=max_depth)
    result = tot.solve(problem, llm_call=_tot_llm_call)
    if result.get("error"):
        return f"ToT 推理失败: {result['error']}"

    chain = result.get("chain", [])
    answer = result.get("answer", "")
    calls = result.get("llm_calls", 0)

    output_parts = [f"[ToT 推理完成，共调用 LLM {calls} 次]"]
    if chain:
        output_parts.append("推理路径:")
        for i, step in enumerate(chain, 1):
            output_parts.append(f"  Step {i}: {step}")
    if answer:
        output_parts.append(f"结论: {answer}")

    return "\n".join(output_parts)


_tot_llm_call = None


def set_tot_llm_call(func):
    """设置 ToT 内部使用的 LLM 调用函数——由 react_loop.py 初始化时调用"""
    global _tot_llm_call
    _tot_llm_call = func
