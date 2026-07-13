"""dataset — 测试用例加载

从 JSON 文件加载测试用例，每条的字段规范见 dataset.json /
capability_dataset.json。
也支持代码中直接传入用例字典列表（兼容旧用法）。
"""

import json
import os
from typing import Optional


_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(_EVAL_DIR, "dataset.json")
CAPABILITY_DATASET = os.path.join(_EVAL_DIR, "capability_dataset.json")

CAPABILITIES = (
    "accuracy",
    "tool_selection",
    "reasoning",
    "consistency",
    "hallucination",
)


class TestCase:
    """一条测试用例

    功能验证字段：
        id, question, expected_tools, must_contain, must_contain_any,
        max_steps, timeout, tag, tags, description

    能力评估可选字段：
        capability: accuracy|tool_selection|reasoning|consistency|hallucination
        expected_answer: 标准答案（子串匹配）
        expected_tool_sequence: 有序工具序列
        reasoning_checkpoints: 中间推理要点
        consistency_runs: 重复执行次数（默认 1）
        forbid_claims: 幻觉哨兵（答案中出现则记为幻觉）
        require_grounded: 答案中的数字须能在工具观察/must_contain 中找到依据
    """

    __test__ = False  # 避免 pytest 把本类当测试收集

    def __init__(self, data: dict):
        self.id: str = str(data.get("id", ""))
        self.question: str = data.get("question", "")
        self.description: str = data.get("description", "")
        self.expected_tools: list[str] = data.get("expected_tools", [])
        # 工具别名组：每组至少一个命中即可，例如 [["get_time","get_current_time"],["calculator"]]
        self.expected_tool_groups: list[list[str]] = [
            list(g) for g in (data.get("expected_tool_groups") or [])
        ]
        self.must_contain: list[str] = data.get("must_contain", [])
        self.must_contain_any: list[str] = data.get("must_contain_any", [])
        self.max_steps: int = data.get("max_steps", 10)
        self.timeout: int = data.get("timeout", 60)
        self.tag: Optional[str] = data.get("tag", None)
        self.tags: list[str] = list(data.get("tags", []) or [])

        # 能力评估字段
        self.capability: Optional[str] = data.get("capability")
        self.expected_answer: str = data.get("expected_answer", "") or ""
        self.expected_tool_sequence: list[str] = list(
            data.get("expected_tool_sequence", []) or []
        )
        self.reasoning_checkpoints: list[str] = list(
            data.get("reasoning_checkpoints", []) or []
        )
        self.consistency_runs: int = int(data.get("consistency_runs", 1) or 1)
        self.forbid_claims: list[str] = list(data.get("forbid_claims", []) or [])
        self.require_grounded: bool = bool(data.get("require_grounded", False))

        if self.capability and self.capability not in self.tags:
            self.tags.append(self.capability)
        if self.capability and not self.tag:
            self.tag = self.capability

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "question": self.question,
            "description": self.description,
            "expected_tools": self.expected_tools,
            "must_contain": self.must_contain,
            "must_contain_any": self.must_contain_any,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "tag": self.tag,
            "tags": self.tags,
        }
        if self.capability:
            d["capability"] = self.capability
        if self.expected_tool_groups:
            d["expected_tool_groups"] = self.expected_tool_groups
        if self.expected_answer:
            d["expected_answer"] = self.expected_answer
        if self.expected_tool_sequence:
            d["expected_tool_sequence"] = self.expected_tool_sequence
        if self.reasoning_checkpoints:
            d["reasoning_checkpoints"] = self.reasoning_checkpoints
        if self.consistency_runs > 1:
            d["consistency_runs"] = self.consistency_runs
        if self.forbid_claims:
            d["forbid_claims"] = self.forbid_claims
        if self.require_grounded:
            d["require_grounded"] = self.require_grounded
        return d


def resolve_dataset_path(name_or_path: Optional[str] = None) -> str:
    """解析数据集路径。

    支持：
      - None / "default" / "functional" → dataset.json
      - "capability" / "capabilities" → capability_dataset.json
      - 其它字符串 → 当作文件路径
    """
    if not name_or_path or name_or_path in ("default", "functional"):
        return DEFAULT_DATASET
    if name_or_path in ("capability", "capabilities"):
        return CAPABILITY_DATASET
    return name_or_path


def load_dataset(path: Optional[str] = None) -> list[TestCase]:
    """从 JSON 文件加载测试用例列表

    path 可为：
      - None / "default" / "functional" → dataset.json
      - "capability" → capability_dataset.json
      - 具体文件路径
    """
    filepath = resolve_dataset_path(path)
    if not os.path.exists(filepath):
        print(f"[Eval] 数据集文件不存在: {filepath}，返回空列表")
        return []
    with open(filepath, encoding="utf-8") as f:
        raw_list = json.load(f)
    return [TestCase(item) for item in raw_list]


def filter_by_tag(cases: list[TestCase], tag: str) -> list[TestCase]:
    """按 tag 筛选测试用例"""
    return [c for c in cases if c.tag == tag or tag in c.tags]


def filter_by_capability(
    cases: list[TestCase], capability: str
) -> list[TestCase]:
    """按 capability 筛选。capability='all' 时返回所有带 capability 字段的用例。"""
    if capability in ("all", "*", ""):
        return [c for c in cases if c.capability]
    return [c for c in cases if c.capability == capability]
