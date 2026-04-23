# Agent Decisions

- Date: 2026-01-25
  Decision: Create new skill `algorithm-article-writer` for long-form algorithm explanations with runnable code.
  Rationale: Existing `leetcode-acers-blogger` is specialized for LeetCode; algorithm concept posts need a different structure and language selection rules.
  Scope: `.codex/skills/algorithm-article-writer/` with template, acceptance criteria, and reinforcement scaffolding.

- Date: 2026-02-28
  Decision: Add a source-agnostic skill `algorithm-problem-acers-blogger` and keep `leetcode-acers-blogger` as a backward-compatible LeetCode entry.
  Rationale: The repository has two stable workflows: algorithm problem writeups (OJ/custom) and algorithm concept articles. LeetCode-specific trigger should remain compatible while enabling Codeforces/AtCoder/Luogu/custom problem posts.
  Scope: New file `.codex/skills/algorithm-problem-acers-blogger/SKILL.md`; update `.codex/skills/leetcode-acers-blogger/SKILL.md`.

- Date: 2026-04-23
  Decision: Retire `algorithm-problem-acers-blogger` and `leetcode-acers-blogger` without compatibility wrappers.
  Rationale: The repository now uses a clearer three-stage architecture: `algorithm-tutorial-builder` for derivation, `acers-blog-formatter` for publication structure, and `tech-post-enhancer` for optional strengthening. Keeping the old one-shot ACERS skills would duplicate responsibilities and preserve the wrong mental model.
  Scope: Remove `.codex/skills/algorithm-problem-acers-blogger/` and `.codex/skills/leetcode-acers-blogger/`; keep the three-stage skills as the canonical workflow.
