# Algorithm Article Template (Long-Form, High-Density, Master-Level)

> Fill every section. Keep signal density high: each paragraph should add a new idea, tradeoff, or application detail.
> Include expert signals: abstraction, bounds intuition, decision criteria, and implementation realities.
> Each major section must include at least one concrete anchor (number/constraint/formula/counterexample).

## Title
- <Clear, keyword-rich, specific>

## Subtitle / Summary
- <1-2 sentences: value + target reader>

## Target Audience
- <Beginner/Intermediate/Advanced + who benefits>

## Background / Motivation
- <Why this algorithm matters, where it fails without it. Include a measurable bottleneck.>

## Quick Mastery Map (60-120s)
- Problem shape:
- Core idea in one line:
- When to use / avoid:
- Complexity headline (with n or d):
- Common pitfall (with example or failure case):

## Master Mental Model
- Core abstraction (the invariant or structure you are really exploiting):
- Problem family it belongs to (e.g., interval DP, monotonic structure, flow, greedy-exchange):
- Isomorphism to a known template (what it reduces to):

## Core Concepts and Terms
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

## Baseline and Bottleneck
- Naive approach (with complexity):
- Bottleneck and why it fails (quantify if possible):

## Key Observation
- <The turning point that makes the algorithm work>

## Algorithm Steps (Practice Guide)
1. ...
2. ...
3. ...

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

## Runnable Implementation (Language: <X>)
```<language>
# runnable code
```

## Engineering Scenarios
1. Scenario A: background, why it fits, minimal code snippet.
2. Scenario B: background, why it fits, minimal code snippet.
3. Scenario C: background, why it fits, minimal code snippet.

## Content Expansion Knobs (Use if below minimum reading time)
- Add a second worked example with contrasting input scale or distribution.
- Add one micro-derivation with a numeric trace (>= 3 steps).
- Add a quantified tradeoff table (time/space/accuracy).
- Add a second counterexample or failure mode with fix.

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

## Call to Action
- Try it, compare variants, or leave feedback.
