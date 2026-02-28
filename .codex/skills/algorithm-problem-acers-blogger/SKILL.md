---
name: algorithm-problem-acers-blogger
description: v0.1.0 - Create Hugo ACERS posts for algorithm problems across LeetCode/Codeforces/AtCoder/Luogu/custom sources using docs/leetcode_std.md and current timestamps from `date`.
---

# Algorithm Problem ACERS Blogger

## Trigger
Use when the user provides an algorithm problem (online judge or custom) and wants a publishable ACERS post. This covers LeetCode and non-LeetCode problem writeups. Do not use for broad concept articles or paper reviews.

## Workflow
1. Read `docs/leetcode_std.md` and follow its ACERS template plus all extra requirements.
2. Gather inputs:
   - problem statement and constraints,
   - 1-2 examples,
   - `problem_source` (`leetcode`, `codeforces`, `atcoder`, `luogu`, or `custom`),
   - target language (`zh` or `en`),
   - optional output path override, title/slug/tags/keywords.
3. Normalize `problem_source` to lowercase kebab-case.
4. Choose output path:
   - If user provides path, honor it.
   - Else if `problem_source=leetcode`: `content/<lang>/alg/leetcode/<slug>.md`.
   - Else if `content/<lang>/dev/algorithm/` exists: `content/<lang>/dev/algorithm/<slug>.md`.
   - Else: `content/<lang>/alg/<problem_source>/<slug>.md`.
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
   - Include naive-to-optimized thought process and correctness reasoning.
   - Include practical steps, runnable examples, FAQs, best practices, meta info, and CTA.
7. For algorithm problems, append multi-language implementations (Python, C, C++, Go, Rust, JS) unless user requests a subset.
8. Validate:
   - No invented constraints, inputs, outputs, or complexity claims.
   - Required ACERS sections are present.
   - Front matter is valid and taxonomy is consistent with target folder.
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
