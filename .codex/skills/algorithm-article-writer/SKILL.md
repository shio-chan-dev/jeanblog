---
name: algorithm-article-writer
description: Write long-form, high-density algorithm explanation posts for this Hugo blog with runnable code, using docs/std.md and a project template.
---

# Algorithm Article Writer

## Trigger
Use when the user requests an algorithm explanation post (concept, technique, or series) focused on fast mastery with runnable code. Do not use for LeetCode problem writeups or paper reviews.

## Workflow
1. Read `docs/std.md`, `assets/algorithm-article-template.md`, `references/depth-checklist.md`, `references/reading-time-estimator.md`, and `references/deepening-ladder.md`.
2. Gather required inputs: algorithm/topic, target audience, target language (or infer), code-language constraints, output path override, and any examples/constraints.
3. Choose output path:
   - If the algorithm is AI/ML-specific and a relevant folder exists under `content/<lang>/ai/`, use `content/<lang>/ai/<topic>/<slug>.md`.
   - Else if `content/<lang>/dev/algorithm/` exists: `content/<lang>/dev/algorithm/<slug>.md`.
   - Else if `content/<lang>/alg/` exists: `content/<lang>/alg/<slug>.md`.
   - Otherwise: `content/posts/<category>/<slug>.md`.
   - Keep ASCII kebab-case filenames; preserve slug once chosen.
4. Choose code language using `references/language-selection-rubric.md`.
   - If ambiguous, ask; otherwise pick the best-fit language and record the assumption.
5. Outline using the template; ensure every section from `docs/std.md` is covered.
6. Choose 1-2 core concepts to deepen; list them explicitly in the outline.
7. Draft a long-form, high-density article with master-level structure:
   - At least one runnable code snippet (no pseudocode-only solutions).
   - At least one worked example (input/output or trace).
   - Naive-to-optimized reasoning path and tradeoffs.
   - Correctness reasoning (proof sketch or invariant).
8. Run a deepening pass for the chosen concepts using `references/deepening-ladder.md`.
9. Run an anchor pass using `references/depth-checklist.md`; add missing numeric examples, constraints, formulas, or counterexamples.
10. Run an anti-fluff rewrite: remove generic phrasing and replace with concrete, testable statements.
11. Compute reading time using `scripts/estimate_reading_time.py` and the rules in `references/reading-time-estimator.md`.
   - If estimated minutes < `min_required` (default 15), deepen the chosen core concepts further; do not add unrelated parallel topics.
   - Set `readingTime` to the computed estimate (rounded up); it must never be lower than the estimate.
12. Fill YAML front matter:
   - `title`, `subtitle`, `date`, `summary`, `tags`, `categories`, `keywords`, `readingTime`, `draft`.
   - Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for `date`.
   - Target `readingTime` >= 15 minutes unless the user requests shorter.
13. Validate with `references/acceptance-criteria.md` and fix gaps.
14. Report output (path, date, notes, checks).

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
- Do not invent constraints, inputs, or results; ask when missing.
- Keep taxonomy consistent; do not create new categories without approval.
- Do not edit `themes/`, config files, or generated outputs.
- Use ASCII filenames by default.
- No secrets or PII.
- Every major section must include at least one concrete anchor as defined in `references/depth-checklist.md`.
- `readingTime` must be >= the computed estimate from `scripts/estimate_reading_time.py`.
- If the estimate is below the minimum threshold, deepen the chosen core concepts (do not add unrelated parallel topics).

## Verification
- Front matter valid and required fields present.
- Required sections from `docs/std.md` are present.
- Code snippet is runnable and minimal.
- References/links resolve.

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
