---
name: tutorial-reviewer
description: v0.1.0 - Review teaching-first tutorial drafts for guided-build quality, structural correctness, and incremental code growth when the user wants a pass/fail verdict plus revision targets rather than a rewrite.
---

# Tutorial Reviewer

## Trigger
Use when the user wants a tutorial draft reviewed as a tutorial, not rewritten from scratch. Typical prompts include:

- "Review this tutorial draft."
- "Check whether this is really guided-build."
- "Tell me whether this passes the teaching bar."
- "Find the structural problems before I publish."

Use it for both:

- LeetCode / OJ / one-problem tutorials
- algorithm / method / data-structure tutorials that are still supposed to teach by construction

Do not use it for SEO polish, final blog enhancement, or line-edit rewriting.

## Bundled Resources
- `AGENTS.md` for repo constraints and reporting expectations.
- `references/review-checklist.md` for fail conditions and family-specific review items.
- `references/report-format.md` for the required verdict layout.

## Workflow
1. Read `AGENTS.md`, `references/review-checklist.md`, and `references/report-format.md`.
2. Identify the tutorial family before judging:
   - `problem tutorial`: one concrete statement with inputs/outputs/constraints
   - `algorithm tutorial`: one concrete method, data structure, or architecture taught by derivation
3. Review structure before wording.
   - Opening discipline
   - guided-build chain
   - code-growth continuity
   - final full-code placement
4. Fail fast on structural errors.
   - If the draft opens with concept glossary, background lecture, component preview, or formula sheet before the first tiny task or pressure point, record it as a top issue.
   - If numbered steps exist but the code does not actually grow from one version to the next, record it as a top issue.
5. Check step integrity.
   - Each step should solve one concrete problem when possible.
   - Each step should add or replace one core state, rule, helper, or module.
   - Each step should say what the current version can do and what it still lacks.
6. Check late-step discipline.
   - Do not let Step 5 or Step 7 degrade into architecture talk only.
   - If a later step introduces a real module, block, or class, that step should show the current integrated code for that unit, not just a 2-3 line local fragment or constructor stub.
7. Check final-code discipline.
   - The first full runnable code should appear only when the build has earned it.
   - Do not accept duplicate full-code endings unless the user explicitly asked for a platform wrapper.
   - If a separate `Reference Answer` exists, it must not introduce new logic.
8. Produce a fixed-format review using `references/report-format.md`.
   - Give `Pass` or `Fail`.
   - List issues in severity order.
   - For each issue, cite concrete evidence with file/section/line references when possible.
   - For each issue, give one direct revision target.
9. If the draft passes, still note residual risks or thin areas instead of pretending it is perfect.

## Required Inputs
- A tutorial draft, file path, or pasted body.
- Enough context to tell whether it is a problem tutorial or an algorithm tutorial.

## Defaults
- review mode: structure-first
- verdict mode: binary `Pass` / `Fail`
- evidence mode: cite concrete sections or file lines when available
- rewrite policy: do not rewrite the whole tutorial unless the user explicitly asks after the review
- priority order: structure > teaching flow > code growth > completeness > polish

## Output Format
- Follow `references/report-format.md` exactly.

## Guardrails
- Do not act like an enhancer or copy editor when the core issue is tutorial structure.
- Do not silently rewrite the draft instead of reviewing it.
- Do not bury structural failures under positive summary language.
- Do not mark a draft `Pass` if the opening still front-loads concepts, formulas, or component previews before the first tiny task or pressure point.
- Do not mark a draft `Pass` if steps are numbered but code fragments are still disconnected.
- Do not mark a draft `Pass` if late steps introduce new modules without showing the integrated current code for those modules.
- Do not mark a draft `Pass` if a second full-code or reference section introduces new logic.
- Do not spend the review on SEO, CTA, tags, or metadata unless the user explicitly asked for that review.

## Verification
- Confirm the draft type was identified correctly.
- Confirm the opening starts from a tiny task, trace, or pressure point when the tutorial claims to be guided-build.
- Confirm the draft shows a real growth path rather than a retrospective explanation of a known answer.
- Confirm code fragments genuinely feed the next version.
- Confirm the first full code appears at the right time.
- Confirm the review output contains a binary verdict, critical issues, and concrete revision targets.
