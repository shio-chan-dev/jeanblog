# Acceptance Criteria

## Required Checks

- The post lives under the correct `content/<lang>/thoughts/thoughts/` path unless the user explicitly requested another location.
- Hugo front matter includes:
  - `title`
  - `date`
  - `draft`
  - `summary`
  - `tags`
  - `categories`
  - `keywords`
  - `readingTime`
- The opening includes a conclusion-first thesis.
- The article explains both:
  - what the evaluated thing is good for
  - what it is not good for
- The article states how the evaluated thing relates to the current workflow or baseline.
- The article provides at least one practical decision aid or evaluation method for readers.
- Claims that are not verified are written as judgments or open questions, not as facts.
- The article aligns with the writing checklist in `docs/std.md`.

## Failure Cases

- The piece reads like a setup tutorial instead of an evaluation.
- The comparison never names the incumbent workflow.
- The article praises or dismisses the tool without conditions.
- The article implies a new source of truth without naming the boundary.
