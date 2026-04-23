# Required Review Output Format

Use this structure exactly.

```md
# Verdict
Pass | Fail

# Scope
- Tutorial type: `<problem tutorial | algorithm tutorial>`
- Target checked: `<file path or short identifier>`

# Critical Issues
1. `<severity>: <short issue title>`
   - Evidence: `<section / line / concrete symptom>`
   - Why it fails: `<why this breaks guided-build quality>`
   - Required revision: `<one direct fix target>`

# Strengths
- `<what already works and should be preserved>`

# Residual Risks
- `<what is still thin even if the draft passes>`
```

Rules:

- If there are no critical blockers, write `None.` under `# Critical Issues`.
- Keep the verdict binary.
- Lead with failures, not encouragement.
- Do not rewrite the full tutorial inside the review.
