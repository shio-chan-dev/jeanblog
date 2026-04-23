# Acceptance Criteria (Algorithm Tutorial Builder)

## Required Content
- Title is clear and keyword-rich.
- The first teaching section after front matter starts from one tiny task, trace, or concrete pressure point.
- If target audience/background are present, they do not appear before the first tiny task.
- If core concepts and terms are present, they do not appear as an upfront glossary before the first build steps.
- A derivation section is present and shows how the method is built from the pressure, limitation, or missing capability before naming the final trick.
- The derivation section does not collapse into a one-problem OJ tutorial.
- When code materially helps, the tutorial includes one final runnable complete implementation, end-to-end module, or minimal complete demo.
- Explanation/principles and correctness reasoning are included.
- Opening summary does not preview the full later component list.
- No standalone `naive idea`, `naive approach`, or `naive-to-optimized` section appears; any such contrast is embedded inside the guided-build steps.
- Terms, formulas, and module names are introduced at first real use rather than front-loaded in a preview section.
- If earlier code fragments are used, they clearly feed into the final implementation/demo rather than ending in duplicate assembled/reference code sections.
- If a later step introduces a new module, class, or block, that step shows the integrated current version of that unit instead of only a tiny local fragment or constructor stub.
- Complexity analysis is included (time and space).
- Worked example or trace is included.
- Common pitfalls/edge cases are covered.
- Best practices are listed.
- Summary includes at least 4 concrete takeaways.
- References/further reading are provided.
 - Each major section includes at least one concrete anchor (number/constraint/formula/counterexample).
 - At least one explicit limitation or counterexample is included.
 - Deepening ladder applied to 1-2 named core concepts (PDKH).

## Front Matter
- YAML front matter includes: title, date, draft, tags, categories.
- Slug and output path follow project taxonomy.

## Quality Bar
- No filler; each section adds new insight or tradeoff.
- Code is runnable and minimal.
- Links and assets resolve.
- No secrets or PII.
- The tutorial object is an algorithm or method, not one concrete OJ problem solution.

## Evidence to Record
- Path, date, and any assumptions.
- Taxonomy choices.
- What checks were run (or "not run" and why).
