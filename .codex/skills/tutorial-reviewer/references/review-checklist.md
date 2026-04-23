# Tutorial Review Checklist

Use this checklist to judge whether a tutorial is genuinely guided-build or only
wearing guided-build language.

## Verdict Rule

- `Pass`: no structural blockers remain; only minor improvements or local
  thin spots remain.
- `Fail`: one or more structural blockers remain.

## Structural Blockers

Any of the following is enough to fail the draft:

1. The first teaching section opens with target-audience prose, background
   lecture, concept glossary, component preview, or formula summary before the
   first tiny task or pressure point.
2. The draft explains an already-known answer instead of building toward it.
3. Steps are numbered, but code fragments are not actually incremental.
4. A large bridge is crossed without an intermediate version when one is
   clearly needed.
5. A late step introduces a new module or block but only shows a local fragment
   or constructor stub while the integrated unit appears only in the final code.
6. A separate full-code or reference section introduces new logic.

## Core Guided-Build Checks

Check all of these:

1. Does the tutorial start from a tiny task, trace, or pressure point?
2. Is the pressure concrete enough to make the missing mechanism feel
   necessary?
3. Is the partial state or current representation explained in plain language?
4. Is the smaller remaining subproblem explicit when recursion or staged growth
   is involved?
5. Is the completion condition explicit?
6. Are the next choices or next additions explicit?
7. Are choose / update / undo or carry-forward rules explicit when relevant?
8. Does each step add or replace only one core thing when possible?
9. After each step, does the tutorial say what the current version can do now
   and what it still lacks?
10. Do code fragments genuinely feed into the next version?
11. Does the first full code appear only after the build has earned it?
12. If a `Reference Answer` exists, does it avoid adding new logic?

## Problem Tutorial Checks

Use these when the object is one concrete OJ-style problem.

- The tutorial starts from problem evidence, not a named template.
- A smaller subproblem is stated explicitly.
- There is a first correct version before the optimized version.
- Optimization stages are preserved when the bridge is non-trivial.
- The tutorial does not ask "why do we need X?" before defining `X`.

## Algorithm Tutorial Checks

Use these when the object is one concrete algorithm, method, architecture, or
data structure.

- The opening tiny task exposes a real missing capability, not just a topic.
- Terms, formulas, and module names appear at first real use, not in an upfront
  glossary.
- Later steps stay incremental instead of reverting to architecture overview.
- If implementation matters, the build converges to one final runnable complete
  implementation or minimal complete demo.

## Severity Guide

- `Critical`: the draft fails as a guided-build tutorial.
- `Major`: the draft may still teach something, but the build chain is weak or
  broken in an important place.
- `Minor`: the structure passes, but one local section is thinner or less clear
  than it should be.
