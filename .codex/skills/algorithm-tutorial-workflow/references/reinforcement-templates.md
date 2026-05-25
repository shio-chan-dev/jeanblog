# Reinforcement Templates

## JSONL Record Format
Each line in `references/reinforcement-audit.jsonl` is a single JSON object with a `step` and required fields.

### Plan Record
```
{"step":"plan","timestamp":"YYYY-MM-DDTHH:MM:SSZ","objective":"...","acceptance_criteria":"...","scope_in":"...","scope_out":"...","inputs":"...","exit_condition":"..."}
```

### Change Record
```
{"step":"change","timestamp":"YYYY-MM-DDTHH:MM:SSZ","failure_mode":"...","edits":"...","why":"...","expected_shift":"...","rollback":"..."}
```

### Verify Record
```
{"step":"verify","timestamp":"YYYY-MM-DDTHH:MM:SSZ","checks_run":"...","evidence":"...","metrics_snapshot":"...","decision":"..."}
```

### Reflect Record
```
{"step":"reflect","timestamp":"YYYY-MM-DDTHH:MM:SSZ","improvements":"...","what_worked":"...","risks_tradeoffs":"...","next_action":"...","outcome":"..."}
```

## Step Templates

Plan:
```
Objective:
Acceptance criteria:
Scope in:
Scope out:
Inputs:
Exit condition:
```

Change:
```
Failure mode targeted:
Edits:
Why:
Expected shift (before/after):
Rollback:
```

Verify:
```
Checks run:
Evidence:
Metrics snapshot:
Decision:
```

Reflect:
```
Improvements:
What worked:
Risks/tradeoffs:
Next action:
Outcome:
```
