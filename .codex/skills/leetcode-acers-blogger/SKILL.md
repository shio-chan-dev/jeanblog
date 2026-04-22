---
name: leetcode-acers-blogger
description: v0.1.6 - Create publishable Hugo ACERS posts for concrete LeetCode problems when the user wants a LeetCode-specific guided-build tutorial that starts from the problem block and grows code step by step from a minimal correct base.
---

# LeetCode ACERS Hugo Blogger

## Trigger
Use when the user explicitly asks for a LeetCode problem writeup, a Hot100 article tied to a LeetCode problem, or names this skill. Use it when the user wants a publishable Hugo post that teaches the solution as a LeetCode-specific guided build rather than a terse reference answer. If the problem source is not LeetCode, switch to `algorithm-problem-acers-blogger`.

## Bundled Resources
- `.codex/skills/algorithm-problem-acers-blogger/SKILL.md` as the delegated primary workflow.
- `.codex/skills/algorithm-problem-acers-blogger/references/source-path-policy.md` for LeetCode and Hot100 placement policy.
- `.codex/skills/algorithm-problem-acers-blogger/references/derivation-first-tutorial.md` for the “teach it from scratch” walkthrough mode.
- `.codex/skills/leetcode-acers-blogger/references/leetcode-guided-build-quality.md` for LeetCode-specific teaching order and anti-pattern checks.
- `.codex/skills/algorithm-problem-acers-blogger/references/verification-checklist.md` for final article validation.
- `docs/leetcode_std.md` for the ACERS structure.

## Workflow
1. Set `problem_source=leetcode` by default.
2. Read `.codex/skills/algorithm-problem-acers-blogger/SKILL.md`, `.codex/skills/leetcode-acers-blogger/references/leetcode-guided-build-quality.md`, and the delegated references needed for path policy and verification.
3. Gather the supplied problem statement, constraints, examples, target language, and any path override. If the user provides only partial LeetCode text, use only the supplied facts and surface unknowns instead of inventing missing constraints.
4. Keep default output path locked to:
   - Chinese: `content/zh/alg/leetcode/<slug>.md`
   - English: `content/en/alg/leetcode/<slug>.md`
   - If the user explicitly requests Hot100 or the task context clearly targets an existing Hot100 collection, use the matching `content/<lang>/alg/leetcode/hot100/...` path.
   - If user specifies another path, honor it.
5. Keep default category `LeetCode` unless user explicitly requests a different taxonomy.
6. Build the article body in this order:
   - problem block first,
   - then the guided-build section,
   - then `Assemble the Full Code`,
   - then `Reference Answer`,
   - then the rest of the ACERS sections.
7. In the guided-build section, force the teaching chain to grow from a minimal correct base:
   - start with the smallest runnable skeleton or first stable state,
   - state what each new step adds relative to the previous code,
   - let the first complete version be correct even if it is naive,
   - introduce helper state such as `used`, `cols`, `diag1`, `diag2`, `prefix`, or `dp` only after the bottleneck in the naive version is visible,
   - define a concept before asking a “why do we need X?” question about it.
8. Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for front matter date.
9. Run both the delegated verification checklist and the LeetCode-specific guided-build quality checks before delivery.
10. Report output path, date, source, assumptions, and checks.

## Required Inputs
- Problem statement (full text or pasted core constraints).
- 1-2 examples.
- Target language (`zh` or `en`).
- Output folder override (optional).

## Defaults
- `problem_source`: `leetcode`
- `draft`: `false`
- default category: `LeetCode`
- default collection: standard LeetCode path, not `hot100/`, unless the user or task context clearly requires Hot100
- teaching mode: guided build from minimal base to optimized solution

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Source: `leetcode`
- Notes: `<assumptions or missing info>`
- Checks: `<validation run>`

## Guardrails
- Do not invent constraints, inputs, or outputs.
- Follow `docs/leetcode_std.md` ACERS requirements.
- Use runnable code and include reasoning path, not final answer only.
- Start LeetCode problem posts with the problem block before any `Target Readers`, `Background / Motivation`, or standalone `Core Concepts` section.
- Do not include secrets or private data.
- Do not move a LeetCode post into Hot100 unless the request or task context clearly targets that collection.
- Do not skip the guided-build section and jump straight to the named template or final code.
- Do not generate a standalone `naive idea` / `naive-to-optimized` section; any necessary contrast must live inside the numbered guided-build steps.
- Do not ask a question about a helper structure before defining the underlying concept it models.
- Do not introduce optimized helper state before showing the pain point in the simpler correct version.
- Do not let adjacent code fragments feel disconnected; each step must say what changed from the previous step.
- Do not make the `Reference Answer` introduce new logic that never appeared in the guided build.

## Verification
- Confirm the delegated skill still sees `problem_source=leetcode`.
- Confirm the final path stays under `content/<lang>/alg/leetcode/` unless the user or task explicitly targets `hot100/`.
- Confirm the final category remains `LeetCode` unless the user requested a different taxonomy.
- Confirm the problem statement, examples, and constraints appear before any `Target Readers`, `Background / Motivation`, or standalone `Core Concepts` section.
- Confirm the tutorial section is a numbered step-by-step build that leads into `Assemble the Full Code` and `Reference Answer`.
- Confirm the guided build starts from a minimal correct base rather than from pre-optimized helper state.
- Confirm each helper state or optimization is justified by a visible bottleneck in the previous step.
- Confirm concept definition appears before “why do we need X?” style prompting.
