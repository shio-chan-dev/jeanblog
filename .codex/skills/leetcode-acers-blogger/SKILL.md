---
name: leetcode-acers-blogger
description: v0.1.0 - Compatibility entry for LeetCode ACERS posts; delegates to algorithm-problem-acers-blogger with `problem_source=leetcode`.
---

# LeetCode ACERS Hugo Blogger (Compatibility)

## Trigger
Use when the user explicitly asks for LeetCode problem writeups or names this skill. If the problem source is not LeetCode, switch to `algorithm-problem-acers-blogger`.

## Workflow
1. Set `problem_source=leetcode` by default.
2. Read `.codex/skills/algorithm-problem-acers-blogger/SKILL.md` and follow that workflow.
3. Keep default output path locked to:
   - Chinese: `content/zh/alg/leetcode/<slug>.md`
   - English: `content/en/alg/leetcode/<slug>.md`
   - If user specifies another path, honor it.
4. Keep default category `LeetCode` unless user explicitly requests a different taxonomy.
5. Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for front matter date.

## Required Inputs
- Problem statement (full text or pasted core constraints).
- 1-2 examples.
- Target language (`zh` or `en`).
- Output folder override (optional).

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Source: `leetcode`
- Notes: `<assumptions or missing info>`

## Guardrails
- Do not invent constraints, inputs, or outputs.
- Follow `docs/leetcode_std.md` ACERS requirements.
- Use runnable code and include reasoning path, not final answer only.
- Do not include secrets or private data.
