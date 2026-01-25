# Depth Checklist (Anti-Fluff)

Each major section must contain at least one concrete anchor. A concrete anchor is one of:
- A numeric example (with actual numbers or sizes)
- A constraint boundary (e.g., n <= 1e5, memory <= 8GB)
- A formula with variable definitions
- A counterexample or failure case
- A tradeoff quantified (time/space cost or accuracy)

## Section Requirements
- Background/Motivation: name a real bottleneck or measurable limitation.
- Quick Mastery Map: include one numeric complexity or threshold and one failure mode.
- Master Mental Model: link to a known template and cite a specific invariant.
- Core Concepts: provide at least one formula with defined variables.
- Feasibility/Lower Bound: state a limit and a case where it breaks.
- Problem Framing: specify input size range and objective metric.
- Baseline/Bottleneck: include a simple baseline with complexity.
- Key Observation: tie to the invariant and what it removes.
- Steps: include at least one explicit intermediate state or transformation.
- Selection Guide: include at least two regime splits with numbers.
- Complexity: include both time and space, and dominant term.
- Engineering Realities: mention one concrete optimization and its risk.
- Alternatives/Tradeoffs: include at least one quantified tradeoff.
- Pitfalls: include at least one failure case example.
- Summary: each takeaway must be actionable.
- Reading time: compute estimate and ensure it is not below the threshold.
- Deepening: identify 1-2 core concepts and apply the PDKH ladder steps to each.
