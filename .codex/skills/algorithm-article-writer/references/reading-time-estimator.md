# Reading Time Estimator (Strict, Minimum-Only)

## Goal
Compute a conservative estimate of reading time. The `readingTime` field must be >= the estimate. Exceeding the estimate is allowed.

## Tokenization Rules
- Chinese characters: count every Han character as 1 unit.
- English words: split on whitespace/punctuation, count tokens of [A-Za-z0-9]+.
- Code blocks: count word tokens in fenced code blocks separately (stricter than prose).
- Math blocks: count each display math block as 12 tokens (≈ 12 words).

## Default Speeds (Tune if you have data)
- Chinese: 560 chars/min
- English: 470 words/min
- Code: 150 words/min
- Math blocks: 12 tokens each (pre-weighted)

## Estimate Formula
```
minutes = ceil(
  cn_chars / 560
  + en_words / 470
  + code_words / 150
  + math_blocks * (12 / 470)
)
```

## Policy
- If `minutes < min_required` (default 15), expand content until `minutes >= min_required`.
- Never set `readingTime` below `minutes`.
- If you cannot reach the minimum without padding, add meaningful material:
  - extra worked example
  - additional tradeoff analysis
  - failure case or counterexample
  - one more engineering scenario
