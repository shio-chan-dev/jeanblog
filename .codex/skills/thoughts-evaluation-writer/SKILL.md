---
name: thoughts-evaluation-writer
description: v0.1.0 - Write publishable jeanblog thoughts posts for tool, workflow, and system evaluations when the user wants a conclusion-first comparison with fit/not-fit boundaries and a clear personal judgment instead of a tutorial.
---

# Thoughts Evaluation Writer

## Trigger
Use when the user wants to turn a real trial, comparison, or workflow judgment
into a publishable `thoughts` post for this Hugo blog. Typical prompts include:

- "Turn this tool trial into a blog post."
- "Write a publishable article comparing this system with my current workflow."
- "Record what this tool is good for and not good for."

Do not use for step-by-step tutorials, algorithm tutorials, LeetCode writeups, or
pure factual explainers without a judgment or comparison angle.

## Bundled Resources
- `AGENTS.md` for project-level writing and safety constraints.
- `docs/std.md` for the house checklist of required blog sections.
- `assets/thoughts-evaluation-template.md` for the default structure.
- `references/path-and-taxonomy-policy.md` for path, category, and slug rules.
- `references/comparison-axes.md` for evaluation dimensions and organization.
- `references/claim-boundary-checklist.md` for verified-fact vs judgment control.
- `references/acceptance-criteria.md` for final validation.

## Workflow
1. Read `AGENTS.md`, `docs/std.md`, `assets/thoughts-evaluation-template.md`, and all files in `references/`.
2. Gather the minimum inputs:
   - subject under evaluation
   - what was actually tried or observed
   - the incumbent workflow, tool, or baseline being compared against
   - target language (`zh` or `en`) or infer from the request
   - whether the user wants a draft or an immediately publishable post
3. Confirm the article type fits this skill.
   - If the main job is "teach readers how to do X", do not use this skill.
   - If the main job is "here is my judgment after trying X", continue.
4. Choose the output path with `references/path-and-taxonomy-policy.md`.
   - Default to `content/<lang>/thoughts/thoughts/<slug>.md`.
   - Keep ASCII kebab-case filenames.
5. Write the thesis first, before the outline.
   - The thesis must answer:
     - what the evaluated thing is good for
     - what it is not good for
     - how it relates to the author's current workflow
     - whether the recommendation is strong, conditional, or negative
6. Select 3-5 comparison axes from `references/comparison-axes.md`.
   - When comparing against an incumbent system, prefer point-by-point comparison by axis.
   - Avoid block writing that first fully explains one side and only later reveals the comparison.
7. Build the outline with `assets/thoughts-evaluation-template.md`.
   - Keep the opening conclusion-first.
   - Include explicit `fit`, `not fit`, and `boundary` sections.
   - Include at least one practical decision aid for readers.
8. Draft the article with full Hugo front matter:
   - `title`, `subtitle`, `date`, `summary`, `tags`, `categories`, `keywords`, `readingTime`, `draft`
   - Use `categories: ["thoughts"]` unless the user explicitly wants another taxonomy.
9. Run the claim-boundary pass with `references/claim-boundary-checklist.md`.
   - Separate verified behavior from personal inference.
   - Mark uncertainty instead of guessing.
10. Run the final validation pass with `references/acceptance-criteria.md`.
11. If the user asked to publish, set `draft: false`; otherwise leave `draft: true`.
12. Report the result with path, thesis, publish state, assumptions, and checks.

## Required Inputs
- The tool, workflow, or system being evaluated.
- The real experience, comparison, or conclusion to record.
- The current workflow or baseline used for comparison.
- Target language (`zh` or `en`) or permission to infer.
- Publish state (`draft` or publish now) if the user already knows it.

## Defaults
- Output path: `content/<lang>/thoughts/thoughts/<slug>.md`
- Category: `thoughts`
- Language: same as the user request unless told otherwise
- Tone: practical, bounded, conclusion-first, and explicit about tradeoffs
- Publish state: `draft: true` unless the user explicitly asks to publish
- Comparison method: point-by-point by axis
- Title pattern: lead with the conclusion, not the feature list

## Output Format
- Path: `<file path>`
- Thesis: `<1-2 sentence judgment>`
- Publish State: `draft | published`
- Notes: `<assumptions, gaps, or open questions>`
- Checks: `<validation run or not run>`

## Guardrails
- Do not turn an evaluation post into an installation or setup tutorial unless the user explicitly asks for one.
- Do not present a personal trial as a universal fact.
- Do not blur the boundary between formal project source-of-truth docs and personal/tooling reflections.
- Do not recommend a tool without stating the conditions where the recommendation holds.
- Do not omit the "not fit" side of the judgment.
- Do not create new categories or move outside the `thoughts` lane without approval.
- Do not invent current tool behavior; verify with the user's evidence, local repo context, or cited official docs when needed.
- Do not publish a post that lacks a clear thesis in the opening.

## Verification
- Front matter is valid and complete.
- The opening states the conclusion clearly.
- The article includes explicit `fit`, `not fit`, and `boundary` sections.
- The comparison is organized by axis when an incumbent workflow is involved.
- Claims are tagged mentally as verified fact, personal inference, or open question.
- The structure still satisfies the checklist in `docs/std.md`.
