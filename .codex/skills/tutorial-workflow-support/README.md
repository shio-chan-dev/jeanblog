# Tutorial Workflow Support

This package contains support skills for jeanblog tutorial production. These
skills do not replace the LeetCode or algorithm tutorial workflows. They add
local process artifacts around those workflows so long tutorial work can resume
cleanly after context changes.

Runtime artifacts are written under `.agent-runs/` by default. That directory
is ignored by Git because plan records, sketches, and check reports are local
execution state, not publishable blog content.

## Skills

| Phase | Skill | Purpose |
| --- | --- | --- |
| Record | [`tutorial-plan-record`](tutorial-plan-record/SKILL.md) | Save an approved tutorial plan into a local run directory. |
| Sketch | [`tutorial-sketch`](tutorial-sketch/SKILL.md) | Turn a saved plan into a teaching skeleton and code-growth contract. |
| Check | [`tutorial-check`](tutorial-check/SKILL.md) | Compare a draft against the saved plan and sketch before review. |

## Recommended Flow

For LeetCode or OJ problem tutorials:

```text
$leetcode-tutorial-plan
-> $tutorial-plan-record
-> $tutorial-sketch
-> $leetcode-tutorial-build
-> $tutorial-check
-> $leetcode-tutorial-review
```

For standalone algorithm or data-structure tutorials:

```text
$algorithm-tutorial-plan
-> $tutorial-plan-record
-> $tutorial-sketch
-> $algorithm-tutorial-build
-> $tutorial-check
-> $algorithm-tutorial-review
```

Use these support skills when the work will span multiple turns, multiple
agents, or enough time that the plan may otherwise disappear into chat history.
