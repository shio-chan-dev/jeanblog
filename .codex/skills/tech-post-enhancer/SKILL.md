---
name: tech-post-enhancer
description: v0.2.0 - Enhance a stable technical post when the core tutorial and minimal publishable structure are already in place and the user now wants stronger SEO, supporting sections, title polish, multi-language code, or engineering add-ons.
---

# Tech Post Enhancer

## Trigger
Use when a post is already structurally sound and the next goal is to make it stronger, richer, or more publishable without rewriting its core teaching logic. Typical triggers include SEO polish, CTA, better section names, more code languages, stronger summaries, FAQs, engineering applications, or extension sections. This stage assumes the post already has its core derivation, minimal publishable front matter, and final runnable code/demo in place.

## Bundled Resources
- `docs/std.md` for general blog quality expectations.
- `docs/leetcode_std.md` when the post is an algorithm problem writeup.

## Workflow
1. Identify which enhancement goals are actually requested:
   - SEO and metadata,
   - stronger title/subtitle,
   - CTA and takeaway polish,
   - multi-language code completion,
   - engineering applications,
   - FAQs or extension reading.
2. Preserve the core teaching order and only enhance around it.
3. Apply the requested high-value additions in this order:
   - metadata and positioning,
   - clarity and section naming,
   - supporting sections,
   - optional breadth additions such as more languages or scenarios.
4. Keep enhancements evidence-based:
   - runnable code only,
   - concrete engineering examples,
   - no invented claims about performance or real-world usage.
5. Stop when the requested enhancement level is reached; do not inflate the post with filler.
6. Report what was enhanced and what was intentionally left unchanged.

## Required Inputs
- A stable post draft.
- The enhancement goals or target outcomes.
- Target language (`zh` or `en`) if the draft does not already make it clear.

## Defaults
- preserve core tutorial and structure
- prefer high-signal additions over broad expansion
- do not add new sections unless they materially improve the post
- stage policy: enhancement happens after tutorial building, not before

## Output Format
- Enhancement Scope: `<requested upgrades>`
- Changes Applied: `<high-level summary>`
- Notes: `<assumptions, skipped enhancements, unresolved gaps>`

## Guardrails
- Do not rewrite the core derivation unless explicitly asked.
- Do not add fluff, generic inspiration, or repetitive summary paragraphs.
- Do not add engineering scenarios or extra code languages with fake or non-runnable content.
- Do not silently change taxonomy or file placement.
- Do not automatically invoke tutorial-building skills.
- Do not use enhancement work to compensate for a tutorial, derivation, or structure layer that is still weak.

## Verification
- Confirm the enhanced post keeps the same teaching backbone.
- Confirm each added section serves a concrete user-facing purpose.
- Confirm all added code blocks are runnable examples rather than pseudocode-only filler.
- Confirm the enhancement layer did not bloat the article with redundant content.
- Confirm the post was already structurally publishable before enhancement began.
- Confirm the post already had its core derivation and final runnable code/demo before enhancement began.
