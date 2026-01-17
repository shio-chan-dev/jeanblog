---
name: leetcode-acers-blogger
description: Create Hugo blog posts for LeetCode-style problems in this project using docs/leetcode_std.md, with current timestamps from `date`. Use when the user provides a problem statement and wants a new ACERS post.
---

# LeetCode ACERS Hugo Blogger

## Workflow
1. Read `docs/leetcode_std.md` and follow the ACERS template plus all extra requirements.
2. Gather inputs: problem statement, examples, target language (zh/en), target folder override (if any), desired slug/title, tags/keywords.
3. Choose output path:
   - Chinese default: `content/zh/alg/leetcode/<slug>.md`
   - English default: `content/en/alg/leetcode/<slug>.md`
   - Do not default to `content/posts/leetcode`.
   - If the user specifies a path (e.g., `@content/posts/leetcode`), honor that path.
4. Generate front matter in YAML with `title`, `date`, `draft=false`, `categories`, `tags`, `description`, `keywords`.
5. Get the current timestamp by running `date "+%Y-%m-%dT%H:%M:%S%:z"` and use it in `date`.
6. Write the full article in Chinese technical style, including:
   - Title, subtitle/summary, target readers, background/motivation, core concepts
   - ACERS sections (Algorithm, Concepts, Engineering, Reflection, Summary)
   - Practical steps, runnable examples, explanations, FAQs, best practices
   - Meta info (reading time, tags, SEO keywords, meta description) and CTA
7. If it is an algorithm problem, append multi-language implementations (Python, C, C++, Go, Rust, JS).
8. Use ASCII kebab-case file names by default; avoid non-ASCII unless explicitly requested.

## Required Inputs
- Problem statement (full text or link with pasted text)
- 1-2 examples (input/output or scenario)
- Target language (zh or en)
- Output folder (optional; default is the language section under `content/zh/alg/leetcode`)
- Title/keywords/tags (optional; infer if not provided)

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Notes: `<assumptions or missing info>`

## Guardrails
- Do not invent constraints, inputs, or outputs; ask if missing.
- Follow `docs/leetcode_std.md` exactly; do not omit required sections.
- Use runnable code, no pseudocode.
- Keep front matter consistent with existing LeetCode posts.
- Do not include secrets or private data.
