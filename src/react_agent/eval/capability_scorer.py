"""capability_scorer — 能力评估规则打分

指标：
  accuracy         最终答案是否命中 expected_answer / must_contain
  tool_selection   工具精确率 / 召回率 / F1（可选顺序匹配）
  reasoning        推理检查点 + 最终答案
  consistency      同题多次运行答案一致率
  hallucination    禁止主张 + 可选 grounded 检查

无 capability 字段的用例回退到 scorer.score_result（功能 4 维分）。
"""

from __future__ import annotations

import re
from typing import Optional

from .scorer import score_result


def extract_final_answer(stdout: str, trajectory: Optional[dict]) -> str:
    """从 stdout / trajectory 提取最终答案文本。"""
    if trajectory:
        fa = (trajectory.get("final_answer") or "").strip()
        if fa:
            return fa
    m = re.search(r"(?:最终答案|FINAL ANSWER)\s*[:：]\s*(.*)", stdout or "",
                  re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return (stdout or "").strip()


_INFRA_FAIL_MARKERS = (
    "LLM 初始化失败",
    "LLM调用失败",
    "配置文件不存在",
    "[Eval] 超时",
    "[Eval] 执行异常",
)


def _is_infra_failure(text: str) -> bool:
    t = text or ""
    return any(m in t for m in _INFRA_FAIL_MARKERS)


def extract_tools(stdout: str, trajectory: Optional[dict]) -> list[str]:
    """按调用顺序列出工具名（可含重复）。"""
    ordered: list[str] = []
    if trajectory:
        for step in trajectory.get("steps", []):
            if "action" in step and step["action"].get("name"):
                ordered.append(step["action"]["name"])
            for action in step.get("actions", []):
                if action.get("name"):
                    ordered.append(action["name"])
    if not ordered:
        ordered = re.findall(r"\[调工具\] (\w+)\(", stdout or "")
    return ordered


def extract_observations(trajectory: Optional[dict]) -> str:
    """拼接所有工具观察文本，供 grounded 检查。"""
    if not trajectory:
        return ""
    parts = []
    for step in trajectory.get("steps", []):
        obs = step.get("observation")
        if obs:
            parts.append(str(obs))
        if "action" in step and step["action"].get("result"):
            parts.append(str(step["action"]["result"]))
        for action in step.get("actions", []):
            if action.get("result"):
                parts.append(str(action["result"]))
    return "\n".join(parts)


def extract_thoughts(trajectory: Optional[dict], stdout: str = "") -> str:
    """拼接思考文本。"""
    parts = []
    if trajectory:
        for step in trajectory.get("steps", []):
            t = step.get("thought")
            if t:
                parts.append(str(t))
    if not parts and stdout:
        parts.append(stdout)
    return "\n".join(parts)


def normalize_answer(text: str) -> str:
    """规范化答案以便一致性比较。"""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    t = t.replace(",", "").replace("，", "")
    # 去掉常见前缀
    for prefix in ("最终答案:", "finalanswer:", "答案:", "答:"):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t


def answer_matches(answer: str, expected: str) -> bool:
    if not expected:
        return True
    a = normalize_answer(answer)
    e = normalize_answer(expected)
    return e in a or a in e


def score_capability(
    case,
    stdout: str,
    trajectory: Optional[dict] = None,
    run_results: Optional[list] = None,
) -> dict:
    """按 capability 打分；无 capability 时回退功能评分。

    run_results: consistency 用例的多次运行结果
        [{"stdout": str, "trajectory": dict|None}, ...]
    """
    functional = score_result(case, stdout or "", trajectory)
    capability = getattr(case, "capability", None)

    if not capability:
        return {
            "capability": None,
            "passed": functional["passed"],
            "total": functional["total"],
            "max_score": functional["max_score"],
            "metrics": {},
            "details": functional.get("details", {}),
            "functional": functional,
        }

    dispatch = {
        "accuracy": _score_accuracy,
        "tool_selection": _score_tool_selection,
        "reasoning": _score_reasoning,
        "consistency": _score_consistency,
        "hallucination": _score_hallucination,
    }
    fn = dispatch.get(capability)
    if not fn:
        return {
            "capability": capability,
            "passed": False,
            "total": 0,
            "max_score": 1,
            "metrics": {},
            "details": {"error": {"passed": False, "reason": f"未知 capability: {capability}"}},
            "functional": functional,
        }

    if capability == "consistency":
        result = fn(case, run_results or [{"stdout": stdout, "trajectory": trajectory}])
    else:
        result = fn(case, stdout, trajectory)

    result["capability"] = capability
    result["functional"] = functional
    # 兼容报告里用 total/max_score
    if "total" not in result:
        result["total"] = 1 if result.get("passed") else 0
        result["max_score"] = 1
    return result


def _score_accuracy(case, stdout: str, trajectory: Optional[dict]) -> dict:
    answer = extract_final_answer(stdout, trajectory)
    if _is_infra_failure(answer) or _is_infra_failure(stdout or ""):
        return {
            "passed": False,
            "metrics": {"accuracy": 0.0},
            "details": {
                "accuracy": {
                    "passed": False,
                    "reason": f"基础设施失败: {answer[:120]!r}",
                    "answer": answer[:200],
                }
            },
        }
    expected = getattr(case, "expected_answer", "") or ""
    must = list(getattr(case, "must_contain", []) or [])

    hits = []
    misses = []
    if expected:
        if answer_matches(answer, expected):
            hits.append(expected)
        else:
            misses.append(expected)
    for kw in must:
        if kw in answer or kw in (stdout or ""):
            hits.append(kw)
        else:
            misses.append(kw)

    # 无标准时：有非空答案算通过（弱）
    if not expected and not must:
        passed = bool(answer.strip())
        reason = "无 expected_answer/must_contain，仅检查有答案" if passed else "无答案"
    else:
        passed = len(misses) == 0 and bool(answer.strip())
        reason = f"命中: {hits}" if passed else f"缺失: {misses}; 答案={answer[:80]!r}"

    return {
        "passed": passed,
        "metrics": {"accuracy": 1.0 if passed else 0.0},
        "details": {
            "accuracy": {"passed": passed, "reason": reason, "answer": answer[:200]},
        },
    }


def _score_tool_selection(case, stdout: str, trajectory: Optional[dict]) -> dict:
    actual_list = extract_tools(stdout, trajectory)
    actual_set = set(actual_list)
    expected = list(getattr(case, "expected_tools", []) or [])
    groups = [list(g) for g in (getattr(case, "expected_tool_groups", None) or [])]
    sequence = list(getattr(case, "expected_tool_sequence", []) or [])

    # 无 groups 时：expected_tools 视为「别名 OR 组」——命中任一即可满足该槽位
    # 若同时需要多种工具，请用 expected_tool_groups
    if not groups and expected:
        # 时间工具别名自动归并
        time_aliases = {"get_time", "get_current_time"}
        if set(expected) <= time_aliases:
            groups = [expected]
        elif time_aliases & set(expected):
            others = [t for t in expected if t not in time_aliases]
            groups = [list(time_aliases & set(expected))]
            for t in others:
                groups.append([t])
        else:
            # 默认：每个 expected 工具各自一组（必须全中）
            groups = [[t] for t in expected]

    if not groups and not sequence:
        return {
            "passed": True,
            "metrics": {
                "tool_precision": 1.0,
                "tool_recall": 1.0,
                "tool_f1": 1.0,
            },
            "details": {"tool_selection": {"passed": True, "reason": "无预期工具"}},
        }

    # 召回：有多少组被命中
    hit_groups = 0
    for g in groups:
        if actual_set & set(g):
            hit_groups += 1
    recall = hit_groups / len(groups) if groups else 1.0

    allowed = set()
    for g in groups:
        allowed |= set(g)
    if sequence:
        allowed |= set(sequence)
    # 常见伴随工具：检索后读页、顺带看时间，不视为选错工具
    _companions = {
        "web_search": {"fetch_page", "get_time", "get_current_time"},
        "fetch_page": {"web_search"},
    }
    for t in list(allowed):
        allowed |= _companions.get(t, set())

    # 精确率：实际调用中属于允许集合的比例（额外工具会拉低）
    if actual_set:
        precision = len(actual_set & allowed) / len(actual_set)
    else:
        precision = 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    sequence_ok = True
    if sequence:
        sequence_ok = _is_subsequence(sequence, actual_list)

    # 通过：所需槽位全覆盖；允许少量额外工具（P>=0.25）
    passed = recall >= 1.0 and sequence_ok and (precision >= 0.25 or len(actual_set) <= len(groups) + 1)

    reason = (
        f"actual={actual_list}, groups={groups}, "
        f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        + (f", sequence_ok={sequence_ok}" if sequence else "")
    )
    return {
        "passed": passed,
        "metrics": {
            "tool_precision": round(precision, 3),
            "tool_recall": round(recall, 3),
            "tool_f1": round(f1, 3),
            **({"sequence_ok": sequence_ok} if sequence else {}),
        },
        "details": {
            "tool_selection": {
                "passed": passed,
                "reason": reason,
            }
        },
    }


def _is_subsequence(need: list[str], have: list[str]) -> bool:
    if not need:
        return True
    i = 0
    for name in have:
        if name == need[i]:
            i += 1
            if i == len(need):
                return True
    return False


def _score_reasoning(case, stdout: str, trajectory: Optional[dict]) -> dict:
    answer = extract_final_answer(stdout, trajectory)
    thoughts = extract_thoughts(trajectory, stdout)
    blob = f"{thoughts}\n{answer}\n{stdout or ''}"
    checkpoints = list(getattr(case, "reasoning_checkpoints", []) or [])
    expected = getattr(case, "expected_answer", "") or ""
    must = list(getattr(case, "must_contain", []) or [])

    missing_cp = [c for c in checkpoints if c not in blob]
    answer_ok = True
    if expected:
        answer_ok = answer_matches(answer, expected)
    elif must:
        answer_ok = all(k in answer or k in (stdout or "") for k in must)

    passed = not missing_cp and answer_ok and bool(answer.strip())
    reason = (
        f"checkpoints_ok={not missing_cp}, answer_ok={answer_ok}"
        + (f", missing={missing_cp}" if missing_cp else "")
    )
    return {
        "passed": passed,
        "metrics": {
            "reasoning_pass": 1.0 if passed else 0.0,
            "checkpoint_hit_rate": (
                round((len(checkpoints) - len(missing_cp)) / len(checkpoints), 3)
                if checkpoints else 1.0
            ),
        },
        "details": {
            "reasoning": {"passed": passed, "reason": reason, "answer": answer[:200]},
        },
    }


def _score_consistency(case, run_results: list) -> dict:
    """run_results: [{stdout, trajectory}, ...]"""
    answers = [
        normalize_answer(extract_final_answer(r.get("stdout", ""), r.get("trajectory")))
        for r in (run_results or [])
    ]
    answers = [a for a in answers if a]
    n = len(run_results or [])
    if n <= 1:
        rate = 1.0 if answers else 0.0
        passed = bool(answers)
        reason = f"单次运行，答案={'有' if answers else '无'}"
    elif not answers:
        rate = 0.0
        passed = False
        reason = "多次运行均无答案"
    else:
        # 两两一致比例
        pairs = 0
        agree = 0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                pairs += 1
                if answers[i] == answers[j] or answers[i] in answers[j] or answers[j] in answers[i]:
                    agree += 1
        rate = agree / pairs if pairs else 0.0
        # 也可用众数覆盖率
        from collections import Counter
        most_common_count = Counter(answers).most_common(1)[0][1]
        coverage = most_common_count / len(answers)
        rate = max(rate, coverage)
        passed = rate >= 0.67 and bool(answers)
        # 若有 expected_answer，还要求众数答案匹配
        expected = getattr(case, "expected_answer", "") or ""
        if expected and answers:
            mode = Counter(answers).most_common(1)[0][0]
            if not answer_matches(mode, expected):
                passed = False
        reason = f"runs={n}, unique={len(set(answers))}, consistency_rate={rate:.2f}"

    return {
        "passed": passed,
        "metrics": {"consistency_rate": round(rate, 3)},
        "details": {
            "consistency": {
                "passed": passed,
                "reason": reason,
                "answers": answers[:5],
            }
        },
    }


def _claim_in_text(claim: str, text: str) -> bool:
    """禁止主张匹配：纯数字用词边界，避免 '0' 命中 '7006652'。"""
    if not claim or not text:
        return False
    c = claim.strip()
    if c.isdigit():
        return bool(re.search(rf"(?<!\d){re.escape(c)}(?!\d)", text))
    return c in text


def _score_hallucination(case, stdout: str, trajectory: Optional[dict]) -> dict:
    answer = extract_final_answer(stdout, trajectory)
    blob = f"{answer}\n{stdout or ''}"
    if _is_infra_failure(blob):
        return {
            "passed": False,
            "metrics": {"hallucinated": 1.0, "hallucination_rate": 1.0},
            "details": {
                "hallucination": {
                    "passed": False,
                    "reason": f"基础设施失败，无法评估幻觉: {answer[:120]!r}",
                    "answer": answer[:200],
                }
            },
        }
    forbid = list(getattr(case, "forbid_claims", []) or [])
    hit_forbid = [c for c in forbid if _claim_in_text(c, blob)]

    ungrounded = []
    if getattr(case, "require_grounded", False):
        evidence = extract_observations(trajectory)
        evidence_blob = f"{evidence}\n" + "\n".join(
            getattr(case, "must_contain", []) or []
        )
        # 抽取答案中的较长数字（年份、结果等）
        nums = re.findall(r"\d{2,}", answer)
        for num in nums:
            if num not in evidence_blob and num not in (getattr(case, "expected_answer", "") or ""):
                # 若 expected_answer 含该数字则允许
                expected = getattr(case, "expected_answer", "") or ""
                if num not in expected:
                    ungrounded.append(num)

    hallucinated = bool(hit_forbid) or bool(ungrounded)
    # 无答案也视为失败（无法判定则不算通过）
    if not answer.strip():
        hallucinated = True
        reason = "无最终答案，记为失败"
    elif hit_forbid:
        reason = f"命中禁止主张: {hit_forbid}"
    elif ungrounded:
        reason = f"无依据数字: {ungrounded}"
    else:
        reason = "未发现幻觉哨兵 / 主张可 grounding"

    passed = not hallucinated and bool(answer.strip())
    return {
        "passed": passed,
        "metrics": {
            "hallucinated": 1.0 if hallucinated else 0.0,
            "hallucination_rate": 1.0 if hallucinated else 0.0,
        },
        "details": {
            "hallucination": {
                "passed": passed,
                "reason": reason,
                "answer": answer[:200],
            }
        },
    }
