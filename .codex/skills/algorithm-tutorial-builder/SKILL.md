---
name: algorithm-tutorial-builder
description: v0.2.1 - Build publishable long-form, high-density algorithm and technique tutorials for this Hugo blog when the user wants to understand one concrete algorithm, data structure, or method itself rather than solve one LeetCode or OJ-style problem.
---

# Algorithm Tutorial Builder

## Trigger
Use when the user requests a tutorial about one concrete algorithm, data structure, or method: what it is, why it exists, how it works, how to implement it, and when to use it. Typical examples include Transformer, Union-Find, Segment Tree, Fenwick Tree, A*, Dijkstra, Bloom Filter, or PageRank. Do not use for LeetCode, Hot100, Codeforces, AtCoder, Luogu, or any other one-problem tutorial.

## Bundled Resources
- `docs/std.md` for the project writing checklist.
- `assets/algorithm-tutorial-template.md` for the default tutorial structure.
- `references/derivation-first-explanations.md` for teaching the method from the problem pressure rather than from the final trick.
- `references/language-selection-rubric.md` for code-language selection.
- `references/depth-checklist.md` and `references/deepening-ladder.md` for depth and anti-fluff passes.
- `references/acceptance-criteria.md` for final validation.

## Workflow
1. Read `docs/std.md`, `assets/algorithm-tutorial-template.md`, `references/derivation-first-explanations.md`, `references/depth-checklist.md`, and `references/deepening-ladder.md`.
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
8. Add minimal publishable front matter and repo-aligned taxonomy:
   - `title`, `date`, `draft`, `categories`, `tags`
   - Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for `date`.
   - Default `draft` to `false` unless the user says otherwise.
   - Keep taxonomy consistent with nearby posts in the chosen folder.
   - Keep metadata minimal; do not add SEO keyword lists or computed reading-time fields.
9. Draft a long-form, high-density tutorial with master-level structure:
   - Open the body with one tiny task, worked mini-scenario, or concrete pressure point. Do not open the teaching body with target-audience prose, background overview, concept glossary, or formula summary.
   - At least one runnable code snippet (no pseudocode-only solutions).
   - At least one worked example (input/output or trace).
   - A derivation section that explains how the algorithm or method emerges from the problem pressure, historical limitation, or missing capability.
   - Terms, formulas, and module names must appear at first real use, not as an upfront glossary or preview list.
   - When implementation is part of the learning goal, the tutorial must culminate in one final runnable complete implementation, end-to-end module, or minimal complete demo.
   - If earlier code fragments are used, each fragment must extend the current build and the tutorial must state what the current version can already do and what it still lacks.
   - If a later step introduces a new module, class, or stage boundary, that step must show the current build after the addition, not just a local fragment or constructor stub.
   - Tradeoffs and correctness reasoning.
   - If an earlier approach or wrong instinct matters, use it only to motivate the method itself, not to mimic a LeetCode solution build around one judged input/output task.
   - Correctness reasoning (proof sketch or invariant).
10. Run a deepening pass for the chosen concepts using `references/deepening-ladder.md`.
11. Run an anchor pass using `references/depth-checklist.md`; add missing numeric examples, constraints, formulas, or counterexamples.
12. Run an anti-fluff rewrite: remove generic phrasing and replace with concrete, testable statements.
13. Validate with `references/acceptance-criteria.md` and fix gaps.
14. Report output (path, date, taxonomy, notes, checks).

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
- front matter policy: minimal Hugo front matter only
- output shape: publishable tutorial post
- Tutorial language: same as user request if not specified.
- Code language: chosen via rubric; fallback to Python only if the rubric is inconclusive.

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Taxonomy: `<categories/tags>`
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
- Do not force concept tutorials into LeetCode-style “replace this loop with this loop” problem-solution construction.
- Do use derivation-first growth when code matters: add one mechanism, state, or module at a time and connect it back to the current build.
- Do not start the teaching body with `Target Audience`, `Background`, `Core Concepts`, or a term/formula glossary before the first tiny task or pressure example.
- Do not preview the full component list in the opening summary. The reader should meet modules when the pressure first requires them.
- Do not present the named technique or final formula before the tutorial has shown the bottleneck, limitation, or missing capability that justify it.
- Do not create a standalone `Core Concepts and Terms` or formula section before the first build steps when those same terms will be introduced incrementally later.
- Do not split the teaching flow into redundant “steps”, “implementation”, and “code” sections that restate the same content.
- Do not generate a fake problem-solution ladder just to imitate a tutorial; algorithm tutorials should derive the method, not masquerade as OJ writeups.
- Do not end with duplicated `Assemble the Full Code` / `Reference Answer` sections; the derivation should converge to one final runnable complete implementation or minimal complete demo.
- Do not let late steps collapse back into vague architecture talk; if Step 5 or Step 7 introduces a real new module, show the full current class/block/forward that the reader would actually add at that moment.
- Do not add `readingTime`, `keywords`, CTA sections, or other enhancement-layer metadata by default.
- Every major section must include at least one concrete anchor as defined in `references/depth-checklist.md`.

## Verification
- Confirm the subject is a concrete algorithm, method, or data structure rather than one problem statement.
- Front matter valid and required fields present.
- Required sections from `docs/std.md` are present.
- Final code/demo block is runnable and minimal when implementation is part of the tutorial goal.
- References/links resolve.
- The tutorial derives the algorithm from problem pressure or capability gaps, not from a one-problem solution-construction flow.
- The first teaching section after front matter starts from a tiny task, trace, or concrete pressure rather than audience/background/glossary material.
- Terms, formulas, and module names are introduced at first real use rather than in an upfront preview section.
- If incremental code fragments appear, they clearly feed into the final runnable complete implementation/demo rather than duplicating it.
- If a later step adds a module or block, that step shows the fully integrated current version of that unit, not only an isolated local snippet.
- Confirm the output is publishable Hugo Markdown with minimal front matter.

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
