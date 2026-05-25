# Algorithm Tutorial Template (Long-Form, High-Density, Master-Level)

> Fill every section. Keep signal density high: each paragraph should add a new idea, tradeoff, or application detail.
> Include expert signals: abstraction, bounds intuition, decision criteria, and implementation realities.
> Each major section must include at least one concrete anchor (number/constraint/formula/counterexample).

## YAML Front Matter
```yaml
---
title: "<clear, specific title>"
date: 2026-01-01T00:00:00+08:00
draft: false
categories: ["<existing category>"]
tags: ["<topic-tag-1>", "<topic-tag-2>"]
---
```

## Title
- <Clear, keyword-rich, specific>

## Opening Summary
- <1-2 sentences in the body: what pressure/task the tutorial starts from and what complete build it will reach. Do not preview a component list.>

## Opening Pressure / Tiny Task
- <Start the teaching body here. Use one tiny task, worked mini-scenario, or concrete pressure point.>
- <The reader should feel what is missing before any glossary, audience, or theory section appears.>

## Quick Mastery Map (60-120s)
- <Optional and short. Use only after the opening pressure, never before it.>
- Problem shape:
- Core idea in one line:
- When to use / avoid:
- Complexity headline (with n or d):
- Common pitfall (with example or failure case):

## Target Audience
- <Optional and short. Place only after the opening pressure if it still helps.>

## Background / Motivation
- <Optional and compressed. Keep to a few lines and place only after the opening pressure. Include a measurable bottleneck.>

## Deepening Focus (PDKH Ladder)
- Choose 1-2 core concepts to deepen (name them).
- Apply the PDKH steps from `references/deepening-ladder.md` to each concept.
- No new parallel topics; deepen the same concept through the ladder.

## Master Mental Model
- Core abstraction (the invariant or structure you are really exploiting):
- Problem family it belongs to (e.g., interval DP, monotonic structure, flow, greedy-exchange):
- Isomorphism to a known template (what it reduces to):

## Core Concepts and Terms
- <Optional synthesis section only if the tutorial still needs one later. Do not place this before the first build steps.>
- Definitions:
- Invariants or key properties:
- Data structures:
- Formulae (define variables):

## Feasibility and Lower Bound Intuition
- What cannot be done faster (informal lower bound reasoning with a numeric or regime):
- When input features break the model (counterexample):

## Problem Framing
- Input/Output shape:
- Constraints (if known, include a numeric range):
- Optimization target (time/space/accuracy):

## Derive the Method Step by Step
- Explain how the method emerges from the pressure, bottleneck, or capability gap.
- Use step-by-step growth when code materially helps:
  - Step 1: what fails or is missing without this method?
  - Step 2: what first abstraction, state, or module fixes that?
  - Step 3: what new mechanism must be added next?
  - Step 4: after each addition, what can the current build already do and what does it still lack?
- Introduce each term, formula, and module at first real use inside the relevant step.
- If a later step adds a new module or block, show the current complete version of that unit after the addition. Do not stop at a constructor stub or a 2-3 line local fragment when the real growth point is a full class/block/forward.
- One explicit trace, worked walkthrough, or mechanism example:

## Final Runnable Implementation / Minimal Complete Demo
- End the derivation with one runnable complete implementation, end-to-end module, or minimal complete demo when code materially helps understanding.
- Earlier fragments should converge into this block.
- Do not add a second duplicated `Reference Answer` section after this one.
- Explain only the key integration points, not the whole reasoning again.

## Decision Criteria (Selection Guide)
- Input size regimes (with thresholds):
- Data distribution / sparsity:
- Memory constraints (with limits):
- Implementation complexity tolerance:

## Worked Example (Trace)
- Input (concrete values):
- Step-by-step state (at least 2 steps):
- Output:

## Correctness (Proof Sketch)
- Invariant:
- Why each step preserves it:
- Why termination implies correctness:

## Complexity
- Time (dominant term):
- Space (dominant term):
- Worst/average/best if relevant:

## Constant Factors and Engineering Realities
- Cache/locality considerations (name a bottleneck):
- Practical limits (language/runtime quirks):
- Typical optimizations and their risks (with example):

## Deepening Knobs (Use if the tutorial still feels thin)
- Deepen the chosen concepts with another PDKH step (e.g., formalization or counterexample).
- Add a second worked example only if it is for the same chosen concept.
- Add a quantified tradeoff table only if it compares variants of the same concept.

## Alternatives and Tradeoffs
- Alternative 1 vs current (quantify tradeoff):
- Alternative 2 vs current (quantify tradeoff):
- Why this choice is the most practical:

## Migration Path (Skill Ladder)
- If you master this, next learn:
- How this extends to a harder class of problems:

## Common Pitfalls and Edge Cases
- Pitfall 1 (failure case):
- Pitfall 2 (failure case):
- Pitfall 3 (failure case):

## Best Practices
- ...
- ...
- ...

## Summary / Takeaways
- At least 4 concrete takeaways.

## References and Further Reading
- ...
