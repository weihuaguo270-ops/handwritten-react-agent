"""scorer — Agent 评测评分规则

评分维度：
  1. 工具匹配（Tool Match）— 是否调用了预期工具
  2. 内容匹配（Content Match）— 输出是否包含预期关键词
  3. 步数控制（Step Control）— 是否在预期步数内完成
  4. 答案质量（Answer Quality）— 是否有最终答案（不是空跑）

返回格式：{score, max_score, details: {维度: {passed, detail}}}
"""

import re
from typing import Optional


def score_result(case, stdout: str, trajectory: Optional[dict]) -> dict:
    """对一条测试用例的执行结果打分

    参数:
        case: TestCase 对象（来自 dataset.py）
        stdout: subprocess 捕获的 stdout
        trajectory: 从 trajectories/ 加载的轨迹字典，或 None

    返回:
        {
            "total": int,        # 得分
            "max_score": int,    # 满分
            "passed": bool,      # total == max_score
            "details": {
                "tool_match": {"passed": bool, "reason": str},
                "content_match": {"passed": bool, "reason": str},
                "step_control": {"passed": bool, "reason": str},
                "has_answer": {"passed": bool, "reason": str},
            }
        }
    """
    details = {}
    score = 0

    # 1. 工具匹配（1分）
    tool_detail = _score_tools(case, stdout, trajectory)
    details["tool_match"] = tool_detail
    if tool_detail["passed"]:
        score += 1

    # 2. 内容匹配（1分）
    content_detail = _score_content(case, stdout)
    details["content_match"] = content_detail
    if content_detail["passed"]:
        score += 1

    # 3. 步数控制（1分）
    step_detail = _score_steps(case, trajectory)
    details["step_control"] = step_detail
    if step_detail["passed"]:
        score += 1

    # 4. 有最终答案（1分）
    answer_detail = _score_answer(stdout, trajectory)
    details["has_answer"] = answer_detail
    if answer_detail["passed"]:
        score += 1

    return {
        "total": score,
        "max_score": 4,
        "passed": score == 4,
        "details": details,
    }


def _score_tools(case, stdout: str, trajectory: Optional[dict]) -> dict:
    """工具匹配评分

    从两个来源提取实际调用的工具名：
    1. stdout 中的 [调工具] xxx( 行
    2. trajectory 中的 actions[].name
    """
    # 从 stdout 提取
    called_from_stdout = set(re.findall(r'\[调工具\] (\w+)\(', stdout))
    # 从 trajectory 提取
    called_from_traj = set()
    if trajectory:
        for step in trajectory.get("steps", []):
            if "action" in step:
                called_from_traj.add(step["action"].get("name", ""))
            for action in step.get("actions", []):
                called_from_traj.add(action.get("name", ""))

    actual_tools = called_from_stdout | called_from_traj

    expected = case.expected_tools
    if not expected:
        # 没有预期工具要求→自动通过
        return {"passed": True, "reason": "无预期工具要求"}

    hit = actual_tools & set(expected)
    if hit:
        return {
            "passed": True,
            "reason": f"命中预期工具: {hit}",
        }
    return {
        "passed": False,
        "reason": f"预期工具: {expected}，实际调用: {actual_tools if actual_tools else '(无)'}",
    }


def _score_content(case, stdout: str) -> dict:
    """内容匹配评分

    must_contain: 全部命中才通过
    must_contain_any: 至少命中一个才通过（当 must_contain 为空时使用）
    """
    must_contain = case.must_contain
    must_contain_any = case.must_contain_any

    if not must_contain and not must_contain_any:
        return {"passed": True, "reason": "无预期关键词"}

    if must_contain:
        missing = [k for k in must_contain if k not in stdout]
        if not missing:
            return {"passed": True, "reason": f"命中全部关键词: {must_contain}"}
        # 检查 must_contain_any 作为 fallback
        if must_contain_any:
            hit = [k for k in must_contain_any if k in stdout]
            if hit:
                return {"passed": True, "reason": f"must_contain 缺 {missing}，但 must_contain_any 命中: {hit}"}
        return {"passed": False, "reason": f"缺关键词: {missing}"}

    # 只有 must_contain_any
    hit = [k for k in must_contain_any if k in stdout]
    if hit:
        return {"passed": True, "reason": f"命中关键词: {hit}"}
    return {"passed": False, "reason": f"未命中任何关键词: {must_contain_any}"}


def _score_steps(case, trajectory: Optional[dict]) -> dict:
    """步数控制评分"""
    max_steps = case.max_steps
    if not trajectory:
        return {"passed": False, "reason": "无轨迹数据，无法评估步数"}

    actual_steps = trajectory.get("total_steps", 0)
    if actual_steps == 0:
        actual_steps = len(trajectory.get("steps", []))

    if actual_steps == 0:
        return {"passed": False, "reason": "无步骤记录"}

    if actual_steps <= max_steps:
        return {"passed": True, "reason": f"{actual_steps}/{max_steps} 步"}
    return {"passed": False, "reason": f"{actual_steps}/{max_steps} 步（超限）"}


def _score_answer(stdout: str, trajectory: Optional[dict]) -> dict:
    """答案存在性评分"""
    # 检查 stdout 中有无最终答案标记
    has_final = bool(re.search(r'最终答案', stdout))

    # 检查 trajectory 中有无 final_answer
    has_traj_answer = False
    if trajectory:
        traj_answer = trajectory.get("final_answer", "")
        has_traj_answer = bool(traj_answer and len(traj_answer) > 5)

    if has_final or has_traj_answer:
        return {"passed": True, "reason": "有最终答案"}
    return {"passed": False, "reason": "无最终答案"}
