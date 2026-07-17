"""Harness trajectory schema — validate & normalize Format B JSON.

Canonical schema file: schemas/harness_trajectory.schema.json
Consumers: react-agent recorder, trace-debugger, and llm-eval-engine.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

SCHEMA_FILENAME = "harness_trajectory.schema.json"

# Format B wire version. Bump major on breaking field/semantics changes.
# Trajectories without schema_version are treated as major "1" (compat).
SCHEMA_VERSION = "1"


def schema_major(version: Optional[str] = None) -> str:
    """Return major component of a schema_version string (default: current)."""
    raw = (version if version is not None else SCHEMA_VERSION) or ""
    raw = str(raw).strip()
    if not raw:
        return SCHEMA_VERSION
    return raw.split(".", 1)[0]


def schema_path() -> str:
    """Resolve packaged schema path (repo root /schemas)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # src/react_agent/harness → repo root
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "schemas", SCHEMA_FILENAME)


def load_schema() -> dict:
    path = schema_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TrajectorySchemaError(ValueError):
    """Raised when a trajectory fails Harness Format B validation."""


def validate_trajectory(data: dict, *, strict_one_based: bool = True) -> list[str]:
    """Validate a trajectory dict against Harness Format B rules.

    Returns a list of human-readable issues (empty means OK).
    Does not require the ``jsonschema`` package; checks the critical fields
    that keep agent → tdebug → eval-engine interoperable.
    """
    issues: list[str] = []
    if not isinstance(data, dict) or not data:
        return ["trajectory must be a non-empty object"]

    for key in ("session_id", "query", "steps", "final_answer"):
        if key not in data:
            issues.append(f"missing required field: {key}")

    if "session_id" in data and (
        not isinstance(data["session_id"], str) or not data["session_id"].strip()
    ):
        issues.append("session_id must be a non-empty string")
    if "query" in data and not isinstance(data.get("query"), str):
        issues.append("query must be a string")
    if "final_answer" in data and not isinstance(data.get("final_answer"), str):
        issues.append("final_answer must be a string")

    # Version: absent ⇒ treat as SCHEMA_VERSION major (backward compatible)
    if "schema_version" in data:
        sv = data.get("schema_version")
        if not isinstance(sv, (str, int)) or (isinstance(sv, str) and not sv.strip()):
            issues.append("schema_version must be a non-empty string when present")
        else:
            got = schema_major(str(sv))
            want = schema_major(SCHEMA_VERSION)
            if got != want:
                issues.append(
                    f"schema_version major {got!r} incompatible with supported "
                    f"{want!r} (SCHEMA_VERSION={SCHEMA_VERSION})"
                )

    steps = data.get("steps")
    if steps is not None and not isinstance(steps, list):
        issues.append("steps must be an array")
        return issues
    if not steps:
        issues.append("steps must be a non-empty array")
        return issues

    seen: set[int] = set()
    for i, step in enumerate(steps):
        prefix = f"steps[{i}]"
        if not isinstance(step, dict):
            issues.append(f"{prefix}: must be an object")
            continue
        if "step" not in step:
            issues.append(f"{prefix}: missing 'step' (1-based)")
            continue
        n = step["step"]
        if not isinstance(n, int) or isinstance(n, bool):
            issues.append(f"{prefix}.step: must be an integer")
            continue
        if strict_one_based and n < 1:
            issues.append(
                f"{prefix}.step: must be >= 1 (got {n}); Format B is 1-based"
            )
        if n in seen:
            issues.append(f"{prefix}.step: duplicate step number {n}")
        seen.add(n)

        action = step.get("action")
        actions = step.get("actions")
        if action is not None:
            issues.extend(_validate_tool_call(f"{prefix}.action", action))
        if actions is not None:
            if not isinstance(actions, list) or not actions:
                issues.append(
                    f"{prefix}.actions: must be a non-empty array when present"
                )
            else:
                for j, a in enumerate(actions):
                    issues.extend(_validate_tool_call(f"{prefix}.actions[{j}]", a))

    return issues


def assert_valid(data: dict, *, strict_one_based: bool = True) -> dict:
    """Validate or raise TrajectorySchemaError."""
    issues = validate_trajectory(data, strict_one_based=strict_one_based)
    if issues:
        raise TrajectorySchemaError("; ".join(issues))
    return data


def normalize_trajectory(data: dict) -> dict:
    """Return a shallow-normalized copy suited for cross-repo consumers.

    - Keeps 1-based ``step`` as-is
    - Ensures each tool call exposes string ``arguments`` when only ``args`` exists
    - Flattens singular ``action`` from first of ``actions`` when needed
    - Stamps ``schema_version`` when absent (compat emit)
    """
    out = dict(data)
    if "schema_version" not in out:
        out["schema_version"] = SCHEMA_VERSION
    steps_out = []
    for step in data.get("steps") or []:
        s = dict(step)
        action = s.get("action")
        actions = s.get("actions")
        if action is None and isinstance(actions, list) and actions:
            s["action"] = _normalize_tool_call(actions[0])
            s["actions"] = [_normalize_tool_call(a) for a in actions]
        elif action is not None:
            s["action"] = _normalize_tool_call(action)
            if isinstance(actions, list):
                s["actions"] = [_normalize_tool_call(a) for a in actions]
        steps_out.append(s)
    out["steps"] = steps_out
    if "total_steps" not in out:
        out["total_steps"] = len(steps_out)
    return out


def _normalize_tool_call(action: Any) -> dict:
    if not isinstance(action, dict):
        return {"name": str(action), "arguments": "{}"}
    a = dict(action)
    name = a.get("name") or ""
    a["name"] = str(name)
    args = a.get("arguments", a.get("args"))
    if args is None:
        a["arguments"] = "{}"
    elif isinstance(args, dict):
        a["arguments"] = json.dumps(args, ensure_ascii=False)
        a.setdefault("args", args)
    else:
        a["arguments"] = str(args)
    return a


def _validate_tool_call(prefix: str, action: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(action, dict):
        return [f"{prefix}: must be an object"]
    name = action.get("name")
    if not isinstance(name, str) or not name.strip():
        issues.append(f"{prefix}.name: required non-empty string")
    if "arguments" not in action and "args" not in action:
        return issues
    args = action.get("arguments", action.get("args"))
    if not isinstance(args, (str, dict)):
        issues.append(f"{prefix}: arguments/args must be string or object")
    return issues


def load_and_validate(filepath: str) -> dict:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    assert_valid(data)
    return normalize_trajectory(data)
