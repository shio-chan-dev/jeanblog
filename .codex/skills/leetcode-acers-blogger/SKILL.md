---
name: leetcode-acers-blogger
description: v0.1.4 - Route concrete LeetCode problem writeup requests to the ACERS blogger with `problem_source=leetcode` and guided-build tutorial flow when the user wants a publishable Hugo post.
---

# LeetCode ACERS Hugo Blogger (Compatibility)

## Trigger
Use when the user explicitly asks for LeetCode problem writeups or names this skill. If the problem source is not LeetCode, switch to `algorithm-problem-acers-blogger`.

## Bundled Resources
- `.codex/skills/algorithm-problem-acers-blogger/SKILL.md` as the delegated primary workflow.
- `.codex/skills/algorithm-problem-acers-blogger/references/source-path-policy.md` for LeetCode and Hot100 placement policy.
- `.codex/skills/algorithm-problem-acers-blogger/references/derivation-first-tutorial.md` for the “teach it from scratch” walkthrough mode.
- `docs/leetcode_std.md` for the ACERS structure.

## Workflow
1. Set `problem_source=leetcode` by default.
2. Read `.codex/skills/algorithm-problem-acers-blogger/SKILL.md` and follow that workflow in guided-build mode.
3. Keep default output path locked to:
   - Chinese: `content/zh/alg/leetcode/<slug>.md`
   - English: `content/en/alg/leetcode/<slug>.md`
   - If the user explicitly requests Hot100 or the task context clearly targets an existing Hot100 collection, use the matching `content/<lang>/alg/leetcode/hot100/...` path.
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
- Do not move a LeetCode post into Hot100 unless the request or task context clearly targets that collection.
- Do not skip the guided-build section and jump straight to the named template or final code.
- Do not generate a standalone `naive idea` / `naive-to-optimized` section; any necessary contrast must live inside the numbered guided-build steps.

## Verification
- Confirm the delegated skill still sees `problem_source=leetcode`.
- Confirm the final path stays under `content/<lang>/alg/leetcode/` unless the user or task explicitly targets `hot100/`.
- Confirm the final category remains `LeetCode` unless the user requested a different taxonomy.
- Confirm the tutorial section is a numbered step-by-step build that leads into `Assemble the Full Code` and `Reference Answer`.
