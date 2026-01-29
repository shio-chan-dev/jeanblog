# AGENTS.md (Project Rules)
## Overview
Project: jeanblog. Purpose: store and publish personal blog posts using Hugo + PaperMod; occasional style tweaks.

## Core Principles
- Blog-first: prioritize clear, publishable writing over tooling changes.
- Minimize risk: prefer small, reversible diffs and avoid touching generated outputs.
- Consistency: match existing taxonomy, language, and tone unless asked.
- Transparency: report tests run or explicitly state "not run" with reason.
- Safety: do not introduce secrets, keys, or PII.

## Domain Philosophies (Master-Level)
Omitted domains: Backend, Payments/Billing, ML/AI, Data pipelines, Compliance/Legal, and other non-blog domains are not material to a static personal blog.

### Engineering
- Goal: site builds cleanly and remains maintainable.
- Constraints: avoid hidden coupling; keep config changes minimal.
- Evidence: clean Hugo build or documented "not run" reason.
- Failure Cost: broken site or unreadable content.
- Tradeoffs: prefer clarity over clever customizations.
- Non-negotiables: no breaking changes without explicit approval.

### Design/UX
- Goal: readable, accessible, and pleasant posts.
- Constraints: keep typography and spacing consistent with theme.
- Evidence: clear hierarchy, no layout regressions in preview.
- Failure Cost: readers struggle to consume content.
- Tradeoffs: consistency over visual novelty.
- Non-negotiables: avoid accessibility regressions.

### Education/Docs
- Goal: each post teaches or records a clear idea.
- Constraints: follow a coherent structure and avoid ambiguity.
- Evidence: post includes context, core points, and takeaways.
- Failure Cost: readers cannot learn or reuse the content.
- Tradeoffs: depth can be reduced to improve clarity.
- Non-negotiables: no contradictory guidance inside a post.

### Operations/SRE
- Goal: predictable publishing workflow.
- Constraints: keep deployment config stable.
- Evidence: CI config unchanged unless explicitly required.
- Failure Cost: failed deploys or missing updates.
- Tradeoffs: slower changes to protect stability.
- Non-negotiables: no changes without rollback path.

### Security/Privacy
- Goal: protect secrets and personal data.
- Constraints: no secrets, tokens, or sensitive personal info in repo.
- Evidence: content and config reviewed for leaks.
- Failure Cost: account compromise or privacy harm.
- Tradeoffs: redact rather than expose.
- Non-negotiables: no secret leakage.

## Product & Project Standards
- Primary outcome: publish well-structured blog posts in `content/`.
- Secondary outcome: tasteful, minimal style adjustments via `layouts/` or `static/`.
- Use Hugo front matter per `archetypes/default.md`; keep `draft` accurate.
- Follow the writing checklist in `docs/std.md` for new posts.
- Keep categories/tags consistent with existing menu taxonomy.

## 12 Golden Rules (Why / How / Check)
1. Keep Hugo front matter valid. Why: build and metadata depend on it. How: use `archetypes/default.md` and include title/date/draft/categories/tags. Check: front matter parses and required fields exist.
2. Place posts under `content/zh/<category>/`. Why: site organization and menus. How: create files only under that path. Check: new posts live in `content/zh/`.
3. Do not edit generated outputs. Why: `public/` and `resources/_gen/` are overwritten. How: treat them as read-only unless asked. Check: no edits there.
4. Avoid theme submodule changes by default. Why: high regression risk. How: prefer overrides in `layouts/` and `static/`. Check: no `themes/` changes unless requested.
5. Preserve existing language and tone. Why: consistency for readers. How: keep the post's original language unless asked. Check: no language switches in edits.
6. Keep links and assets valid. Why: broken links hurt credibility. How: verify referenced files exist. Check: all referenced files are present.
7. Use `docs/std.md` as a checklist. Why: maintain post quality. How: include required sections. Check: sections align to checklist.
8. No secrets or PII. Why: security and privacy. How: redact or omit sensitive data. Check: no keys/tokens/PII added.
9. Keep diffs small and scoped. Why: easier review and rollback. How: touch only necessary files. Check: unrelated files unchanged.
10. Explain config/layout changes. Why: prevent silent regressions. How: provide rationale in response. Check: rationale present.
11. Verify build when layout/config changes. Why: avoid broken site. How: run `hugo server -D` or `hugo --minify` if possible. Check: tests reported or "not run" noted.
12. Preserve URLs and slugs. Why: avoid broken links and SEO loss. How: keep filenames and `slug`/`url` unless asked. Check: slugs unchanged.

## Scope Boundaries
- Allowed: all project files.
- Generated outputs: `public/` and `resources/_gen/` are read-only unless explicitly requested.
- Strictly forbidden files/dirs: none. Strictly forbidden actions: adding secrets/keys/PII or rewriting git history.

## Permission Model
- No approval needed for: writing/editing posts in `content/`, updating `docs/`, adjusting `layouts/` or `static/` for minor style tweaks.
- Approval required for: deleting or mass-moving posts, changing `config.toml`/`config.dev.toml`, editing `.github/workflows/`, updating `themes/` submodules, or modifying generated outputs.
- User is the final approver for high-risk changes.

## Execution Rules
- Ask for clarification when requirements are ambiguous.
- Prefer overrides in `layouts/` and `static/` over editing theme files.
- Keep changes reversible and report any assumptions.
- If unsure, pause and ask before proceeding.

## Quality Bar
- Front matter valid and consistent with archetypes.
- Links and assets resolve.
- Tests: run Hugo build/preview for layout/config changes; otherwise note "not run" and why.
- Response includes verification status and any skipped checks.

## Decision & Accountability
- Decision log: `docs/agent-decisions.md` (create if missing when first needed).
- Risk log: `docs/agent-risks.md` (create if missing when first needed).
- Owner: user; agent implements within stated scope.

## Risks & Open Questions
- Highest-risk failure mode: publishing a broken site due to config/theme changes.
- Risk: accidental edits to generated outputs causing confusing diffs.
- Open question: none.
