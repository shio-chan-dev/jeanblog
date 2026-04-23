---
name: acers-blog-formatter
description: v0.1.1 - Turn a stable teaching-first draft into a publishable Hugo ACERS post when the core tutorial logic is already correct and now needs ACERS-compatible blog structure, front matter, taxonomy, and reader-facing formatting.
---

# ACERS Blog Formatter

## Trigger
Use when the tutorial logic is already stable and the next job is to make it publishable in this Hugo blog. Use it for ACERS structure, front matter, taxonomy, section ordering, and reader-facing Markdown cleanup. Do not use it as the first reasoning stage for a problem the model has not yet taught clearly. ACERS is the publication structure handled here, not the reasoning engine handled in the tutorial-building stage.

## Bundled Resources
- `docs/leetcode_std.md` for ACERS algorithm-post requirements.
- `docs/std.md` for general post-quality expectations.
- `archetypes/default.md` for front matter defaults when applicable.

## Workflow
1. Start from an existing tutorial draft or a clearly stable reasoning path.
2. Preserve the teaching chain and reorganize it into a publishable post:
   - title and subtitle,
   - meta information,
   - problem block,
   - ACERS body sections,
   - final checks.
3. Choose the output path and taxonomy from repo conventions:
   - keep algorithm posts under `content/`,
   - prefer existing nearby taxonomy,
   - do not invent new categories without clear evidence.
4. Add or normalize Hugo front matter with valid `title`, `date`, `draft`, `categories`, `tags`, `description`, and `keywords`.
5. Place the tutorial inside the correct reader-facing structure:
   - keep the problem block before `Target Readers`, `Background / Motivation`, or standalone `Core Concepts`,
   - keep the guided build inside `C — Concepts`,
   - keep `Assemble the Full Code` and `Reference Answer` after the step-by-step derivation.
6. Treat ACERS as a publishing shell:
   - keep the tutorial logic intact,
   - do not force a new derivation path just to fit section labels,
   - allow ACERS-compatible wording as long as the post remains structurally publishable in this repo.
7. Clean up headings, tables, emphasis, and code block language tags for publication.
8. Report output path, date, taxonomy choices, and checks run.

## Required Inputs
- A stable tutorial draft or stable reasoning path.
- Target language (`zh` or `en`).
- Desired output path or collection if already known.

## Defaults
- structure: ACERS
- output type: publishable Hugo Markdown
- `draft`: `false` unless the user says otherwise
- formatting priority: preserve teaching flow over cosmetic rewrites
- architecture policy: ACERS is required at the publication layer, not at the tutorial-derivation layer

## Output Format
- Path: `<file path>`
- Date: `<timestamp used>`
- Taxonomy: `<categories/tags>`
- Notes: `<assumptions or unresolved gaps>`
- Checks: `<validation run or not run>`

## Guardrails
- Do not rebuild the reasoning path from scratch if the tutorial logic is still unclear; use the tutorial-building stage first.
- Do not add SEO-heavy or promotional material beyond minimal publishable metadata unless explicitly asked.
- Do not duplicate the same explanation across `Core Concepts`, tutorial steps, and implementation sections.
- Do not move the problem block behind reader-background sections.
- Do not automatically invoke tutorial-building or enhancement skills.
- Do not flatten the guided build just to make the article look more formally ACERS.

## Verification
- Confirm the output is publishable Hugo Markdown with valid front matter.
- Confirm the tutorial chain is preserved rather than flattened into a summary.
- Confirm the problem block appears before audience/background sections.
- Confirm `Assemble the Full Code` and `Reference Answer` remain after the guided build.
- Confirm ACERS was applied as a publication structure rather than as a replacement for the reasoning flow.
