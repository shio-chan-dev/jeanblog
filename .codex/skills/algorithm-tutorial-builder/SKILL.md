---
name: algorithm-tutorial-builder
description: v0.1.1 - Build teaching-first algorithm tutorials when the user wants the solution logic to grow step by step from problem evidence and a minimal correct base before any ACERS formatting, blog packaging, or post enhancement.
---

# Algorithm Tutorial Builder

## Trigger
Use when the user wants to learn or publish an algorithm solution as a true guided build rather than as a polished final article. Use this before blog formatting when the main need is to make the reasoning path teachable and incremental.

## Bundled Resources
- `docs/leetcode_std.md` for algorithm-post section expectations in this repo.
- `.codex/skills/algorithm-problem-acers-blogger/references/derivation-first-tutorial.md` for the default guided-build ladder.

## Workflow
1. Gather only the supplied problem facts: statement, constraints, examples, target language, and any required implementation language.
2. Identify the smallest non-trivial example or bottleneck that exposes why the final method is needed.
3. Build the tutorial around this growth path:
   - problem evidence,
   - smallest runnable skeleton,
   - first state variable,
   - completion condition,
   - first correctness rule,
   - first correct but still naive version,
   - visible bottleneck,
   - optimized helper state or formula,
   - one slow trace,
   - `Assemble the Full Code`,
   - `Reference Answer`.
4. Make each numbered step answer one concrete question and state what changed relative to the previous code.
5. Keep the output teaching-first:
   - prioritize reasoning and code growth,
   - do not require ACERS structure by default,
   - do not add Hugo front matter unless explicitly requested,
   - do not spend tokens on SEO, CTA, or marketing polish.
6. If the problem family is clear, adapt the ladder:
   - backtracking: layer meaning -> choice -> legality -> first correct DFS -> bottleneck -> helper state,
   - DP: state meaning -> base case -> transition -> first correct table/recurrence -> optimization,
   - graph: node/state meaning -> container -> expansion rule -> first correct traversal -> optimization.
7. End with assumptions, missing facts, and what was inferred versus supplied.

## Required Inputs
- Problem statement or faithful summary.
- At least one example.
- Target language for the tutorial (`zh` or `en`).
- Preferred code language if not obvious.

## Defaults
- teaching mode: derivation-first
- default implementation language: follow user request; otherwise prefer Python
- output shape: tutorial draft, not final publishable blog post
- structure policy: ACERS not required at this stage; publication formatting belongs to `acers-blog-formatter`

## Output Format
- Working Title: `<draft title>`
- Scope: `<problem + language + code language>`
- Tutorial Draft: `<guided-build markdown>`
- Notes: `<assumptions, missing facts, inferred pieces>`

## Guardrails
- Do not ask “why do we need X?” before defining what `X` is.
- Do not introduce optimized helper state before the slower correct version exposes the need for it.
- Do not jump from the statement to a named template or final trick in one paragraph.
- Do not force the result into ACERS structure unless the user explicitly asks for that in this stage.
- Do not automatically invoke blog-formatting or enhancement skills.
- Do not invent constraints, examples, or complexity claims.

## Verification
- Confirm the tutorial contains a first correct version before the optimized version.
- Confirm every optimization is justified by a pain point in the previous step.
- Confirm `Assemble the Full Code` does not introduce new logic that never appeared earlier.
- Confirm the output is still a teaching-first draft rather than a packaging-heavy blog post.
- Confirm ACERS-only publishing concerns were left to the formatting stage unless explicitly requested.
