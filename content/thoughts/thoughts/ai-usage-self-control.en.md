---
title: "Do Not Let AI Drive You: Keep the Ability to Build Independently"
subtitle: "Learn from master methods and exercises so AI stays an assistant, not the driver"
date: 2025-12-08
summary: "How to avoid copy-paste dependence when using AI for coding: Feynman technique, deliberate practice, retrieval practice, and a practical self-check workflow."
tags: ["AI assistant", "engineering", "learning", "feynman", "deliberate-practice"]
categories: ["thoughts"]
keywords: ["AI dependence", "self-control", "engineering confidence", "deliberate practice", "Feynman learning"]
readingTime: "Approx. 9 min"
draft: false
---

> Core idea: even with AI, you should be able to implement critical paths offline. AI accelerates; it does not replace thinking. This post combines learning science and practical workflows with a self-checklist.

## Target readers
- Mid to senior engineers and tech leads who want AI speed without losing control.
- Team leads adopting AI-assisted coding or documentation.
- Engineers who already work with Git, tests, and code reviews.

## Background and motivation
- Pain points:
  - Copy-pasting model output without understanding leads to fragile code and hard debugging.
  - Over-reliance on prompts reduces independent implementation ability.
  - Architecture and security decisions get driven by the model instead of the engineer.
- Goals:
  - Implement critical paths from scratch without AI when needed.
  - Use AI for validation and refactoring, not for blind generation.
  - Build a "think first, verify later" workflow.

## Core concepts
- **Feynman technique**: if you can explain it simply, you understand it.
- **Deliberate practice**: target weak points with feedback and challenge.
- **Retrieval practice**: recall and derive before checking answers.
- **Red/blue mode with AI**: human writes first (blue), AI critiques (red).
- **Replaceability**: can you replace the model and still ship the feature?

## Practical steps
1) **Write a human plan first, then ask AI**
   - Sketch interfaces, flow, and edge cases before prompting.
2) **Limit copy/paste; hand-type key logic**
   - Routes, migrations, permissions should be typed by you; AI can review.
3) **Side-by-side comparison**
   - Left: your solution, right: AI suggestions. Keep only what you can explain.
4) **Retrieval practice loop**
   - Implement without AI, then compare with AI, mark blind spots, rewrite once.
5) **Feynman output**
   - Summarize in 3-5 sentences; if you cannot, study again.

## Runnable micro-exercise

Implement a unique function that preserves order:

```python
def unique_keep_order(items):
    seen = set()
    result = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        result.append(x)
    return result

assert unique_keep_order([1, 2, 2, 3]) == [1, 2, 3]
```

Exercise flow:
- Round 1: no AI, implement and test; note gaps.
- Round 2: compare with AI, check edge cases (e.g., unhashable items).
- Round 3: explain complexity and limits to a teammate or in a recording.

## Explanation
- **Why limit copy/paste?**
  - It skips the recall-derive-verify loop and makes understanding shallow.
  - Hand-typing exposes gaps in API knowledge and naming.
- **Trade-offs**
  - Fully manual: safest but slow; use for security-critical modules.
  - AI review: faster but needs human design and merge.
  - AI scaffolding: good for kickstart, but requires tests and refactoring.

## Common questions
- **How to avoid prompt dependence?** Write pseudocode and tests first, then ask AI.
- **What if time is tight?** Ask AI for checklists or tests; you implement the core.
- **How to prove you are not being driven?** Document your decisions and reasons.
- **Security/compliance**: never paste secrets; use local or private models if needed.

## Best practices
- Weekly: rewrite a core path without AI (auth, billing, migrations).
- Add PR template fields: what decisions were made by humans vs AI.
- Use TDD: write tests first, ask AI for edge-case tests only.
- Keep the "explainability" rule: if you cannot explain it in 3 sentences, rework.
- Track blind spots and practice deliberately.

## Conclusion
- AI is a multiplier, not a driver. Keep replaceability and explanation as your safety belt.
- Use Feynman + deliberate practice + retrieval practice to lock in understanding.

## References
- Richard Feynman, "The Feynman Technique"
- Anders Ericsson, "Peak"
- Roediger & Karpicke, "Test-Enhanced Learning"
- Thoughtworks Technology Radar (AI-assisted coding)

## Meta
- Reading time: about 9 minutes
- Tags: AI assistant, engineering practice, learning methods
- SEO keywords: AI dependence, engineering autonomy, deliberate practice, Feynman learning, AI code review
- Updated: 2025-11-14

## Call to Action (CTA)
- Pick one critical module, hand-write it, then use AI to review and record the diff.
- Add an "AI assistance scope" field to your PR template.
- Share your "no-AI rewrite" experiences and learnings.
