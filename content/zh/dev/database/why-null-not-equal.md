---
title: "为什么 SQL 中 NULL 不能用 = 比较：三值逻辑与查询陷阱"
date: 2026-01-24T11:28:00+08:00
draft: false
description: "解释 SQL 三值逻辑下 NULL 的比较规则，并给出正确写法与常见陷阱。"
tags: ["数据库", "SQL", "NULL", "查询", "陷阱"]
categories: ["数据库"]
keywords: ["SQL NULL", "IS NULL", "三值逻辑", "NULL 比较"]
---

## 副标题 / 摘要

在 SQL 中，NULL 代表“未知”，因此 `=` 比较不会返回 true。本文解释三值逻辑的机制，并给出正确写法。

## 目标读者

- 经常写 SQL 的后端工程师
- 在查询结果上踩过 NULL 坑的开发者
- 需要制定查询规范的团队

## 背景 / 动机

很多人会写：

```sql
SELECT * FROM t WHERE field = NULL;
```

然后发现它“不起作用”。原因是 SQL 使用三值逻辑，NULL 不等于任何值（包括 NULL 本身）。

## 核心概念

- **NULL 表示未知**，不是空字符串或 0
- **三值逻辑**：true / false / unknown
- **正确判断方式**：`IS NULL` / `IS NOT NULL`

## 实践指南 / 步骤

1. **判断 NULL 用 `IS NULL`**  
2. **不要用 `=` 与 NULL 比较**  
3. **需要替代值时用 `COALESCE`**  
4. **对外部输入做明确转换**

## 可运行示例

```sql
SELECT id FROM users WHERE deleted_at IS NULL;
```

使用替代值：

```sql
SELECT COALESCE(age, 0) FROM users;
```

## 解释与原理

`NULL = NULL` 的结果是 unknown，而不是 true。  
SQL 的 WHERE 只保留 true 的行，unknown 会被过滤掉，因此查询为空。

## 常见问题与注意事项

1. **NULL 和空字符串是一样吗？**  
   不是，空字符串是确定值。

2. **NULL 参与计算会怎样？**  
   结果通常是 NULL（unknown）。

3. **可以用 `IS DISTINCT FROM` 吗？**  
   部分数据库支持，它能正确处理 NULL。

## 最佳实践与建议

- 团队统一 NULL 处理规范
- 查询中显式处理 NULL
- 在数据模型中明确“缺失 vs 空值”

## 小结 / 结论

NULL 代表未知，因此不能用 `=` 比较。  
理解三值逻辑可以避免大量隐性 bug。

## 参考与延伸阅读

- SQL 标准三值逻辑
- PostgreSQL `IS DISTINCT FROM`

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：SQL、NULL、查询陷阱  
- **SEO 关键词**：SQL NULL, IS NULL, 三值逻辑  
- **元描述**：解释为什么 SQL 中 NULL 不能用 = 比较，并给出正确写法。

## 行动号召（CTA）

检查一次项目里的 SQL，看看有没有 `= NULL` 这种潜在 bug。
