# Source Path Policy

Use this reference when deciding where an algorithm-problem ACERS post should
live and which taxonomy defaults are allowed.

## Path Selection

1. If the user provides an explicit output path, honor it unless it conflicts
   with project rules.
2. Else if `problem_source=leetcode` and the user explicitly requests Hot100,
   or the task context clearly says the post belongs to an existing Hot100
   collection, use `content/<lang>/alg/leetcode/hot100/...`.
3. Else if `problem_source=leetcode`, use
   `content/<lang>/alg/leetcode/<slug>.md`.
4. Else if `content/<lang>/dev/algorithm/` exists and the post is better framed
   as an engineering algorithm note than an online-judge archive entry, use
   `content/<lang>/dev/algorithm/<slug>.md`.
5. Else use `content/<lang>/alg/<problem_source>/<slug>.md`.

## Taxonomy Defaults

- LeetCode paths default to `categories: ["LeetCode"]`.
- `content/zh/dev/algorithm/` defaults to `categories: ["逻辑与算法"]`.
- For other folders, mirror nearby posts in the same folder before inventing a
  new category.
- Keep filenames ASCII kebab-case unless the user explicitly overrides them.

## Hot100 Rule

- Do not silently place LeetCode problems into `hot100/` based only on personal
  judgment that the problem feels "important".
- Require either explicit user intent or clear task context that the Hot100
  collection is the target.
