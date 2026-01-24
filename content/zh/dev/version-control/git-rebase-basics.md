---
title: "什么是 rebase：作用、风险与正确用法"
date: 2026-01-24T11:20:31+08:00
draft: false
description: "解释 Git rebase 的作用、风险与适用场景，并给出安全使用建议。"
tags: ["版本控制", "Git", "rebase", "协作"]
categories: ["版本控制"]
keywords: ["git rebase", "变基", "提交历史", "协作"]
---

## 副标题 / 摘要

rebase 可以让提交历史更线性，但也会重写历史。本文解释它的价值、风险与使用边界。

## 目标读者

- 希望保持整洁提交历史的开发者
- 团队协作中经常处理冲突的人
- 需要制定 Git 规范的技术负责人

## 背景 / 动机

合并分支会产生大量 merge commit，让历史难以阅读。  
rebase 能把分支“挪到”最新主线之上，形成更清晰的线性历史。

## 核心概念

- **rebase**：把分支的提交“搬到”新基线
- **历史重写**：提交哈希会变化
- **交互式 rebase**：整理、合并提交

## 实践指南 / 步骤

1. **仅对本地未推送的分支使用 rebase**  
2. **拉取最新主线再 rebase**  
3. **解决冲突并继续**  
4. **必要时用交互式 rebase 压缩提交**  
5. **公共分支禁止 rebase**

## 可运行示例

```bash
# 在 feature 分支上
git fetch origin

git rebase origin/main

# 若冲突
# 解决后：
git add .
git rebase --continue
```

## 解释与原理

rebase 会“重放”每一个提交到新基线上，因此提交哈希会改变。  
这让历史更整洁，但也意味着共享分支上会引发冲突与丢失提交的风险。

## 常见问题与注意事项

1. **rebase 与 merge 有什么区别？**  
   rebase 改写历史，merge 保留历史分叉。

2. **为什么公共分支不能 rebase？**  
   因为它会改写其他人已基于的历史。

3. **rebase 失败怎么办？**  
   使用 `git rebase --abort` 回滚。

## 最佳实践与建议

- 本地开发分支用 rebase
- 发布分支用 merge
- 设置团队约定，避免误用

## 小结 / 结论

rebase 是整理历史的利器，但必须在正确的边界内使用。  
理解“历史重写”是安全使用的前提。

## 参考与延伸阅读

- Pro Git: Rebasing
- Atlassian Git tutorials

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：Git、rebase、版本控制  
- **SEO 关键词**：git rebase, 变基  
- **元描述**：解释 rebase 的作用、风险与正确用法。

## 行动号召（CTA）

在一个个人分支上尝试 `git rebase -i`，体验整理历史的效果。
