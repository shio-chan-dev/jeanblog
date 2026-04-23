---
name: algorithm-article-writer
description: v0.1.4 - Write long-form, high-density algorithm and technique articles for this Hugo blog when the user wants to understand one concrete algorithm, data structure, or method itself rather than solve one LeetCode or OJ-style problem.
---

# Algorithm Article Writer

## Trigger
Use when the user requests an article about one concrete algorithm, data structure, or method: what it is, why it exists, how it works, how to implement it, and when to use it. Typical examples include Transformer, Union-Find, Segment Tree, Fenwick Tree, A*, Dijkstra, Bloom Filter, or PageRank. Do not use for LeetCode, Hot100, Codeforces, AtCoder, Luogu, or any other one-problem tutorial.

## Bundled Resources
- `docs/std.md` for the project writing checklist.
- `assets/algorithm-article-template.md` for the default article structure.
- `references/derivation-first-explanations.md` for teaching the method from the problem pressure rather than from the final trick.
- `references/language-selection-rubric.md` for code-language selection.
- `references/depth-checklist.md` and `references/deepening-ladder.md` for depth and anti-fluff passes.
- `references/reading-time-estimator.md` plus `scripts/estimate_reading_time.py` for `readingTime`.
- `references/acceptance-criteria.md` for final validation.

## Workflow
1. Read `docs/std.md`, `assets/algorithm-article-template.md`, `references/derivation-first-explanations.md`, `references/depth-checklist.md`, `references/reading-time-estimator.md`, and `references/deepening-ladder.md`.
2. Gather required inputs: algorithm/topic, target audience, target language (or infer), code-language constraints, output path override, and any examples/constraints.
3. Reject problem-solution requests early.
   - If the user provides one concrete problem statement with inputs, outputs, and constraints and the main job is “solve this problem step by step”, use `leetcode-tutorial-builder` instead.
   - Continue only when the object of explanation is the algorithm, method, or system component itself.
4. Choose output path:
   - If the algorithm is AI/ML-specific and a relevant folder exists under `content/<lang>/ai/`, use `content/<lang>/ai/<topic>/<slug>.md`.
   - Else if `content/<lang>/dev/algorithm/` exists: `content/<lang>/dev/algorithm/<slug>.md`.
   - Else if `content/<lang>/alg/` exists: `content/<lang>/alg/<slug>.md`.
   - Otherwise: `content/posts/<category>/<slug>.md`.
   - Keep ASCII kebab-case filenames; preserve slug once chosen.
5. Choose code language using `references/language-selection-rubric.md`.
   - If ambiguous, ask; otherwise pick the best-fit language and record the assumption.
6. Outline using the template; ensure every section from `docs/std.md` is covered.
7. Choose 1-2 core concepts to deepen; list them explicitly in the outline.
8. Draft a long-form, high-density article with master-level structure:
   - At least one runnable code snippet (no pseudocode-only solutions).
   - At least one worked example (input/output or trace).
   - A derivation section that explains how the algorithm or method emerges from the problem pressure, historical limitation, or missing capability.
   - A minimal implementation skeleton or runnable demo when implementation is part of the learning goal.
   - Tradeoffs and correctness reasoning.
   - If an earlier approach or wrong instinct matters, use it only to motivate the method itself, not to mimic a LeetCode solution build.
   - Correctness reasoning (proof sketch or invariant).
9. Run a deepening pass for the chosen concepts using `references/deepening-ladder.md`.
10. Run an anchor pass using `references/depth-checklist.md`; add missing numeric examples, constraints, formulas, or counterexamples.
11. Run an anti-fluff rewrite: remove generic phrasing and replace with concrete, testable statements.
12. Compute reading time using `scripts/estimate_reading_time.py` and the rules in `references/reading-time-estimator.md`.
   - If estimated minutes < `min_required` (default 15), deepen the chosen core concepts further; do not add unrelated parallel topics.
   - Set `readingTime` to the computed estimate (rounded up); it must never be lower than the estimate.
13. Fill YAML front matter:
   - `title`, `subtitle`, `date`, `summary`, `tags`, `categories`, `keywords`, `readingTime`, `draft`.
   - Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for `date`.
   - Target `readingTime` >= 15 minutes unless the user requests shorter.
14. Validate with `references/acceptance-criteria.md` and fix gaps.
15. Report output (path, date, notes, checks).

## Required Inputs
- Algorithm/topic and scope.
- Target audience level (beginner/intermediate/advanced).
- Target language (zh/en) or "infer from request".
- Code language constraints (if any).
- Output path override (optional).

## Defaults
- Output path: `content/<lang>/dev/algorithm/` for non-AI algorithms.
- Category: use existing taxonomy; default to `逻辑与算法` for `content/zh/dev/algorithm/`, otherwise mirror categories from nearby posts in the same folder or ask.
- Tags: include `algorithms` plus topic-specific tags.
- Reading time: long-form (>= 15 min) unless user requests shorter.
- Article language: same as user request if not specified.
- Code language: chosen via rubric; fallback to Python only if the rubric is inconclusive.

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Notes: `<assumptions or missing info>`
- Checks: `<tests run or "not run">`

## Guardrails
- Must include at least one runnable code snippet.
- Do not use this skill for one concrete OJ-style problem tutorial; use `leetcode-tutorial-builder` instead.
- Do not invent constraints, inputs, or results; ask when missing.
- Keep taxonomy consistent; do not create new categories without approval.
- Do not edit `themes/`, config files, or generated outputs.
- Use ASCII filenames by default.
- No secrets or PII.
- Do not force concept articles into LeetCode-style “replace this loop with this loop” incremental code construction.
- Do not present the named technique or final formula before the article has shown the bottleneck, limitation, or missing capability that justify it.
- Do not split the teaching flow into redundant “steps”, “implementation”, and “code” sections that restate the same content.
- Do not generate a fake problem-solution ladder just to imitate a tutorial; concept articles should derive the method, not masquerade as OJ writeups.
- Every major section must include at least one concrete anchor as defined in `references/depth-checklist.md`.
- `readingTime` must be >= the computed estimate from `scripts/estimate_reading_time.py`.
- If the estimate is below the minimum threshold, deepen the chosen core concepts (do not add unrelated parallel topics).

## Verification
- Confirm the subject is a concrete algorithm, method, or data structure rather than one problem statement.
- Front matter valid and required fields present.
- Required sections from `docs/std.md` are present.
- Code snippet is runnable and minimal.
- References/links resolve.
- `readingTime` is not lower than the computed estimate.
- The article derives the algorithm from problem pressure or capability gaps, not from a one-problem solution-construction flow.

## Acceptance Loop
- Run `references/acceptance-criteria.md` and record pass/fail evidence.
- Capture gaps with scope impact and owner.
- Define a next-iteration checklist (highest-impact gap first).
- Name the highest-risk gap and the verification step.

## Reinforcement Plan (disabled by default)
- Enable only with explicit signal: `reinforcement=on`.
- Use templates in `references/reinforcement-templates.md`.
- After each step (plan/change/verify/reflect), prompt "continue?" and wait.
- Log each step to `references/reinforcement-audit.jsonl`.
- Validate with `scripts/validate_reinforcement_audit.py`.
