#!/usr/bin/env python3
import argparse
import json
import sys

REQUIRED_FIELDS = {
    "plan": [
        "step",
        "timestamp",
        "objective",
        "acceptance_criteria",
        "scope_in",
        "scope_out",
        "inputs",
        "exit_condition",
    ],
    "change": [
        "step",
        "timestamp",
        "failure_mode",
        "edits",
        "why",
        "expected_shift",
        "rollback",
    ],
    "verify": [
        "step",
        "timestamp",
        "checks_run",
        "evidence",
        "metrics_snapshot",
        "decision",
    ],
    "reflect": [
        "step",
        "timestamp",
        "improvements",
        "what_worked",
        "risks_tradeoffs",
        "next_action",
        "outcome",
    ],
}


def _is_non_empty_str(value):
    return isinstance(value, str) and value.strip() != ""


def validate_line(obj, line_no, errors):
    if not isinstance(obj, dict):
        errors.append(f"Line {line_no}: not a JSON object")
        return
    step = obj.get("step")
    if step not in REQUIRED_FIELDS:
        errors.append(f"Line {line_no}: invalid or missing step")
        return
    required = REQUIRED_FIELDS[step]
    missing = [key for key in required if key not in obj]
    if missing:
        errors.append(f"Line {line_no}: missing fields: {', '.join(missing)}")
    for key in required:
        if key in obj and key != "step" and not _is_non_empty_str(obj[key]):
            errors.append(f"Line {line_no}: field '{key}' must be a non-empty string")


def main():
    parser = argparse.ArgumentParser(description="Validate reinforcement audit JSONL records.")
    parser.add_argument(
        "path",
        nargs="?",
        default=".codex/skills/algorithm-article-writer/references/reinforcement-audit.jsonl",
        help="Path to reinforcement-audit.jsonl",
    )
    args = parser.parse_args()

    try:
        with open(args.path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        return 2

    errors = []
    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {idx}: invalid JSON ({exc})")
            continue
        validate_line(obj, idx, errors)

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
