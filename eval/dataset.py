"""dataset — 测试用例加载

从 JSON 文件加载测试用例，每条的字段规范见 dataset.json。
也支持代码中直接传入用例字典列表（兼容旧用法）。
"""

import json
import os
from typing import Optional


DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


class TestCase:
    """一条测试用例

    字段：
        id:        唯一标识（用于报表关联）
        question:  输入给 Agent 的问题（必填）
        expected_tools: 预期调用的工具名列表（至少命中一个即通过）
        must_contain:   输出中必须包含的关键词列表（全部命中才通过）
        must_contain_any: 输出中至少命中一个的关键词列表
        max_steps:      最大步数上限
        timeout:        单条用例超时（秒）
        tag:            分类标签（local / web / rag / mcp / agent / orchestrator）
    """

    def __init__(self, data: dict):
        self.id: str = str(data.get("id", ""))
        self.question: str = data.get("question", "")
        self.description: str = data.get("description", "")
        self.expected_tools: list[str] = data.get("expected_tools", [])
        self.must_contain: list[str] = data.get("must_contain", [])
        self.must_contain_any: list[str] = data.get("must_contain_any", [])
        self.max_steps: int = data.get("max_steps", 10)
        self.timeout: int = data.get("timeout", 60)
        self.tag: Optional[str] = data.get("tag", None)
        self.tags: list[str] = data.get("tags", [])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "expected_tools": self.expected_tools,
            "must_contain": self.must_contain,
            "must_contain_any": self.must_contain_any,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "tag": self.tag,
        }


def load_dataset(path: Optional[str] = None) -> list[TestCase]:
    """从 JSON 文件加载测试用例列表

    参数:
        path: dataset.json 路径，默认使用 eval/dataset.json

    返回:
        TestCase 列表
    """
    filepath = path or DEFAULT_DATASET
    if not os.path.exists(filepath):
        print(f"[Eval] 数据集文件不存在: {filepath}，返回空列表")
        return []
    with open(filepath, encoding="utf-8") as f:
        raw_list = json.load(f)
    return [TestCase(item) for item in raw_list]


def filter_by_tag(cases: list[TestCase], tag: str) -> list[TestCase]:
    """按 tag 筛选测试用例"""
    return [c for c in cases if c.tag == tag or tag in c.tags]
