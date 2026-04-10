# Path And Taxonomy Policy

## Default Placement

- Chinese evaluation posts:
  - `content/zh/thoughts/thoughts/<slug>.md`
- English evaluation posts:
  - `content/en/thoughts/thoughts/<slug>.md`

## Use `thoughts` When

- The article records a judgment, comparison, or engineering philosophy.
- The value is in the conclusion, tradeoffs, or boundary-setting.
- The article is driven by a real trial or decision rather than a setup guide.

## Do Not Use `thoughts` When

- The article is mainly a tutorial, setup guide, or operational runbook.
- The article's main job is "teach me how to do it from zero."
- The article belongs to an established technical lane such as algorithm,
  Python, Linux, or Git notes.

## Filename Rules

- Use ASCII only.
- Use lowercase kebab-case.
- Preserve the slug once chosen unless the user explicitly asks to change it.

## Category Rules

- Default `categories: ["thoughts"]`.
- Reuse existing tags when possible; do not create new taxonomy branches just
  to fit one post.
