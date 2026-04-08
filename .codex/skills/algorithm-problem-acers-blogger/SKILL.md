---
name: algorithm-problem-acers-blogger
description: v0.1.3 - Create publishable Hugo ACERS posts for concrete algorithm problems across LeetCode/Codeforces/AtCoder/Luogu/custom sources when the user wants a guided-build tutorial that develops code step by step before the final answer.
---

# Algorithm Problem ACERS Blogger

## Trigger
Use when the user provides an algorithm problem (online judge or custom) and wants a publishable ACERS post. This covers LeetCode and non-LeetCode problem writeups. Do not use for broad concept articles or paper reviews.

## Bundled Resources
- `docs/leetcode_std.md` for the ACERS structure and required sections.
- `references/source-path-policy.md` for source-specific output paths and taxonomy defaults.
- `references/derivation-first-tutorial.md` for the mandatory “build it from scratch” tutorial flow.
- `references/verification-checklist.md` for pre-delivery validation.

## Workflow
1. Read `docs/leetcode_std.md`, `references/source-path-policy.md`, `references/derivation-first-tutorial.md`, and `references/verification-checklist.md`.
2. Gather inputs:
   - problem statement and constraints,
   - 1-2 examples,
   - `problem_source` (`leetcode`, `codeforces`, `atcoder`, `luogu`, or `custom`),
   - target language (`zh` or `en`),
   - optional output path override, title/slug/tags/keywords.
3. Normalize `problem_source` to lowercase kebab-case.
4. Choose output path:
   - Follow `references/source-path-policy.md`.
   - If user provides path, honor it.
   - Use ASCII kebab-case filenames by default.
5. Generate YAML front matter with:
   - `title`, `date`, `draft=false`, `categories`, `tags`, `description`, `keywords`.
   - Use `date "+%Y-%m-%dT%H:%M:%S%:z"` for `date`.
   - Category defaults:
     - LeetCode path: `["LeetCode"]`.
     - `content/zh/dev/algorithm/`: `["逻辑与算法"]`.
     - Other paths: mirror nearby posts in the same folder; if unavailable, ask user before inventing taxonomy.
6. Write the full ACERS article in the target language:
   - Include title/subtitle, target readers, background/motivation, core concepts.
   - Include one mandatory guided-build tutorial section before the final polished algorithm section:
     - write it as numbered `Step 1`, `Step 2`, `Step 3`... moves,
     - make each step answer one question only,
     - make each step add one small code fragment or one precise state rule,
     - introduce variables such as `path`, `used`, `dp`, `queue`, or `prefix` only after explaining why that state is needed,
     - show one slow branch / trace before the full code,
     - then include `Assemble the Full Code`,
     - then include a clean `Reference Answer`.
   - Include naive-to-optimized thought process and correctness reasoning.
   - Include practical steps, runnable examples, FAQs, best practices, meta info, and CTA.
7. For algorithm problems, append multi-language implementations (Python, C, C++, Go, Rust, JS) unless user requests a subset.
8. Validate:
   - Run `references/verification-checklist.md`.
9. Report output path, date, source, assumptions, and checks.

## Required Inputs
- Problem statement (or pasted core text with constraints).
- At least one example.
- Target language (`zh` or `en`).
- `problem_source` (optional; default `leetcode`).
- Output path override (optional).

## Defaults
- `problem_source`: `leetcode`.
- `draft`: `false`.
- File name: ASCII kebab-case.
- Article language: follow user request language.

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Source: `<problem_source>`
- Notes: `<assumptions or missing info>`
- Checks: `<tests run or "not run">`

## Guardrails
- Do not invent problem constraints or examples.
- Keep taxonomy consistent; do not create new categories without approval.
- Use runnable code only; no pseudocode-only final deliverable.
- Do not edit `themes/`, config files, or generated outputs.
- Do not include secrets or private data.
- Do not place a post under `hot100/` unless the user explicitly asks for Hot100 or the task context clearly requires that collection.
- Do not jump directly from the problem statement to the final trick or template label.
- If the final method is a known template, explain why the problem evidence leads to that template before presenting finished code.
- Do not split the teaching flow into three redundant sections that repeat the same content in different words.
- Do not show the full finished code before the guided-build steps and the assembly step are complete.

## Verification
- Front matter is valid and category/tag choices match the selected path.
- Required ACERS sections from `docs/leetcode_std.md` are present.
- Complexity claims, constraints, and examples are traceable to the supplied problem statement.
- Final code blocks are runnable examples rather than pseudocode-only stubs.
- The tutorial path shows how the solution is discovered from the problem, not just what the final solution is.
- The guided-build section is numbered, grows the solution step by step, and leads into `Assemble the Full Code` and `Reference Answer`.
