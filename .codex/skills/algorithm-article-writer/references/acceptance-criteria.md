# Acceptance Criteria (Algorithm Article Writer)

## Required Content
- Title is clear and keyword-rich.
- Subtitle/summary states value and target reader.
- Target audience is explicit.
- Background/motivation explains why the algorithm matters.
- Core concepts and terms are defined.
- Practice guide (step-by-step algorithm) is present.
- At least one runnable code snippet is included.
- Explanation/principles and naive-to-optimized reasoning are included.
- Complexity analysis is included (time and space).
- Worked example or trace is included.
- Common pitfalls/edge cases are covered.
- Best practices are listed.
- Summary includes at least 4 concrete takeaways.
- References/further reading are provided.
- CTA is present.
 - Each major section includes at least one concrete anchor (number/constraint/formula/counterexample).
 - At least one explicit limitation or counterexample is included.
 - Deepening ladder applied to 1-2 named core concepts (PDKH).
 - Expansion for length stays within those concepts; no unrelated parallel topics added.

## Front Matter
- YAML front matter includes: title, subtitle, date, summary, tags, categories, keywords, readingTime, draft.
- readingTime indicates long-form depth (>= 15 min) unless user requested shorter.
- readingTime is >= computed estimate from `scripts/estimate_reading_time.py`.
- Slug and output path follow project taxonomy.

## Quality Bar
- No filler; each section adds new insight or tradeoff.
- Code is runnable and minimal.
- Links and assets resolve.
- No secrets or PII.

## Evidence to Record
- Path, date, and any assumptions.
- What checks were run (or "not run" and why).
