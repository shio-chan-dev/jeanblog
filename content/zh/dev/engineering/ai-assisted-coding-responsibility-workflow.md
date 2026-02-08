---
title: "AI 辅助编程不黑盒：责任主线工作流实战"
date: 2026-02-07T13:16:00+08:00
draft: false
description: "用一套可执行的 commit 与分支流程，把 AI 变成加速器而不是黑盒来源。"
tags: ["AI 编程", "工程实践", "Git 工作流", "DDD", "学习方法"]
categories: ["工程实践"]
keywords: ["AI 辅助编程", "黑盒代码", "责任主线", "commit 策略", "领域建模"]
---

## 副标题 / 摘要

你不需要“每个 commit 都手打重写”，但必须对核心 commit 具备独立实现能力。本文给出一套可落地的 AI 协作流程：让 AI 负责胶水和草稿，你负责领域规则、状态变化与边界裁决。

## 目标读者

- 正在大量使用 AI 写代码，但担心自己变成“黑盒工程师”的开发者
- 想同时提升交付效率和技术判断力的工程师
- 在做 DDD、业务系统或中后台项目的开发者

## 背景 / 动机

AI 代码生成越来越强，这不是坏事。真正的风险在于：当你只会“接收实现”，却不能解释“为何这样实现”，系统一复杂就失去控制权。  
问题不是“要不要用 AI”，而是“哪些决策必须留在人手里”。

## 核心概念

- **责任主线（main）**：只保留你愿意为其设计与后果负责的 commit。
- **AI 草稿（ai/draft-\*）**：一次性候选实现分支，用于对照、压力测试与发现盲区。
- **`git worktree`**：在不二次 clone 的前提下，为不同分支创建多个物理目录（一个仓库，多处并行）。
- **核心 commit**：包含领域规则、不变量、状态机、关键边界与失败路径的提交。
- **胶水 commit**：CRUD、DTO、映射、样板接口、注释等可标准化提交。
- **硬核评价标准**：去掉 AI 和网络，你仍能写出该 commit 的核心伪代码，并解释每一步为什么这么做。

## 实践指南 / 步骤

1. **先分层，再决定谁写**  
   把需求拆成 Domain / Application / Infrastructure。  
   Domain 核心规则优先自己实现；Infrastructure 优先交给 AI。

2. **先写判断标准，再看 AI 方案**  
   先写不变量、边界条件、错误路径、伪代码或测试，再生成 AI 草稿。  
   先有你的“尺子”，再拿 AI 代码来量。

3. **为每轮任务创建短生命周期草稿分支**  
   从当前 `main` 派生 `ai/draft-<topic>`，让 AI 快速给出候选实现。  
   草稿分支只做提议，不做长期维护。

4. **按并行需求选择执行模式**  
   轻量场景：一个目录里切分支即可。  
   高频并行：使用 `git worktree` 给 `main` 与 `ai/draft-*` 各开独立目录。

5. **核心逻辑在 main 重写，不直接拷贝**  
   对核心 commit，避免“在 AI 代码上修修补补”。  
   从你的模型出发重写，再对照吸收 AI 的有价值细节。

6. **胶水层直接复用，提高吞吐**  
   对非核心代码，可以直接 cherry-pick 或复制并快速审查。  
   把脑力留给不可替代的系统判断。

7. **草稿分支通常不 merge，完成后删除**  
   `main` 是责任集合，不是候选方案仓库。  
   只把你认可并能负责的结果放入主线。

## 可运行示例

```bash
# 方案 A：单目录（最小流程）
git switch main
git switch -c ai/draft-order-pricing

# 在草稿分支让 AI 生成候选实现
git add .
git commit -m "ai: draft order pricing implementation"

# 回到主线，按你的模型重写核心逻辑
git switch main

# 对照草稿，不直接 merge
git show ai/draft-order-pricing:src/domain/order_pricing.ts
git diff main..ai/draft-order-pricing -- src/domain/order_pricing.ts

# 提交你负责的实现
git add .
git commit -m "feat: implement order pricing domain logic"

# 删除草稿分支
git branch -D ai/draft-order-pricing
```

```python
class Order:
    def __init__(self, items, status, coupon=None, is_vip=False):
        self.items = items
        self.status = status
        self.coupon = coupon
        self.is_vip = is_vip
        self.final_price = None

    def calculate_price(self):
        if self.status != "CREATED":
            raise ValueError("order status invalid")
        if self.final_price is not None:
            raise ValueError("price already calculated")

        base = sum(item["price"] * item["qty"] for item in self.items)
        discount = 0

        if self.coupon and self.is_vip:
            raise ValueError("coupon and vip discount cannot coexist")
        if self.coupon:
            discount = self.coupon["amount"]
        elif self.is_vip:
            discount = base * 0.05

        self.final_price = max(base - discount, 0)
        return self.final_price
```

```bash
# 方案 B：从 clone 开始的 worktree 并行
git clone git@github.com:you/my-project.git
cd my-project

# 创建 AI 草稿分支（若不存在）
git branch ai/draft-order-pricing

# 为 AI 草稿分支创建第二个目录（不是第二次 clone）
git worktree add ../my-project-ai ai/draft-order-pricing

# 终端 1（你）：main 目录
cd /home/you/my-project
nvim .

# 终端 2（AI）：draft 目录
cd /home/you/my-project-ai
nvim .

# 任一目录都可对照分支差异
git diff main..ai/draft-order-pricing
git diff --name-only main..ai/draft-order-pricing
git diff main..ai/draft-order-pricing -- src/domain/order_pricing.ts

# 结束后清理
cd /home/you/my-project
git worktree remove ../my-project-ai
git branch -D ai/draft-order-pricing
```

## 解释与原理

这套流程的本质是“并行思考，延迟裁决”：

- AI 负责快速给出候选路径和实现样本。
- 你负责定义正确性标准与架构边界。
- 最终 commit 记录的是你的责任判断，而不只是代码结果。

因此你获得的不是“手速训练”，而是“系统可解释、可预测、可修复”的工程能力。

## 例子地图：什么你主导，什么交给 AI

先用一条总规则：

- **规则多、状态复杂、责任重**：你主导。
- **结构固定、可替换、出错代价低**：AI 接管。

### 后端

**你主导（核心）**

- 订单/计费/结算：价格规则、折扣叠加、退款回滚、幂等等关键约束。
- 权限/鉴权/风控：角色模型、越权边界、条件授权。
- 状态机/工作流：订单流转、审批流、补偿逻辑（Saga）。
- 跨服务协调：事务边界、事件顺序、最终一致性。

**AI 接管（胶水）**

- Controller/API 层：参数解析、错误映射、常规路由组装。
- Repository/ORM 层：CRUD、查询拼装、Mapper 样板代码。

### 前端

**你主导（核心）**

- 复杂交互状态：多步骤表单、条件可见性、撤销重做。
- 权限与可见性：角色差异、按钮可用条件、页面访问边界。
- 性能关键路径：大列表、虚拟滚动、缓存策略。

**AI 接管（胶水）**

- 组件样式/布局：CSS、Tailwind、过渡动画与样式细节。
- 页面拼装：普通列表页、Dashboard 拼接与模板化页面。

### 测试

**你主导（核心）**

- 关键性质测试：不变量、边界条件、失败路径。

**AI 接管（增强）**

- 用例扩展：参数组合、随机数据、冗余覆盖补齐。

### 基础设施与 DevOps

**你主导（核心）**

- 架构决策：服务拆分、RPC/消息取舍、一致性模型。
- 安全策略：密钥管理、权限模型、网络隔离边界。

**AI 接管（胶水）**

- CI 配置：标准化流水线模板与任务编排。
- 工程脚本：部署脚本、迁移脚本、重复运维动作。

### 数据与分析

**你主导（核心）**

- 指标定义：业务口径、去重规则、时间窗口。

**AI 接管（实现）**

- SQL 落地：`JOIN`、`GROUP BY`、窗口函数等具体写法。

### AI 系统本身

**你主导（核心）**

- Prompt 契约：输入/输出结构、失败兜底、风险控制。

**AI 接管（实现）**

- Prompt 措辞：示例编写、表达变体、格式润色。

### 快速判断模板

每次动手前先问一句：

> 这段代码是否在定义“必须成立的事实”？

- 如果在定义事实（规则、不变量、边界）→ 你主导。
- 如果在执行事实（搬运、组装、样板）→ AI 接管。

## 常见问题与注意事项

1. **每个 commit 都要自己重写吗？**  
   不需要。只对核心 commit 强制重写，胶水层优先复用。

2. **一定要两个仓库或两个工作区吗？**  
   非必须。多数场景一个目录 + 分支切换足够。  
   只有在你要并行写两个分支时，才建议 `git worktree`（一个仓库，多个目录），不建议维护两个独立 clone。

3. **AI 草稿要不要保留在主线历史里？**  
   通常不要。草稿可留在临时分支或 PR 讨论，不建议 merge 到 `main`。

4. **是否需要不同终端？**  
   不是必须，但强烈建议。  
   一个终端盯 `main`，另一个终端盯 `ai/draft-*`，可以显著降低分支与目录混淆。

5. **可以用 `git diff` 看两个分支差异吗？**  
   可以，而且是最推荐的对照方式：  
   `git diff main..ai/draft-order-pricing`（全量差异）  
   `git diff main..ai/draft-order-pricing -- src/domain/order_pricing.ts`（单文件差异）

6. **如何判断自己是否还在成长？**  
   看能否在断网状态下解释并写出核心 commit 的伪代码与边界处理。

## 最佳实践与建议

- 用“核心 20% 自主 + 胶水 80% AI”作为默认策略。
- 核心逻辑优先测试先行，让测试定义正确性边界。
- commit 信息区分 `ai:`（提议）与 `feat/fix/refactor:`（你负责的决策）。
- 每周做一次复盘：哪些规则你能独立实现，哪些仍是黑盒。

## 小结 / 结论

AI 时代真正稀缺的不是“会不会写代码”，而是“是否能对系统关键判断负责”。  
把 AI 当候选实现生成器，把 `main` 当责任主线，你就能同时获得效率和能力增长。

## 参考与延伸阅读

- Martin Fowler: *Domain-Driven Design Aggregate*
- Martin Kleppmann: *Designing Data-Intensive Applications*
- Git 官方文档：`git worktree`

## 元信息

- **阅读时长**：8~10 分钟  
- **标签**：AI 编程、工程实践、Git 工作流  
- **SEO 关键词**：AI 辅助编程, 黑盒代码, commit 策略  
- **元描述**：一套可执行的 AI 编程工作流，帮你在提升效率的同时保持对核心逻辑的掌控。

## `worktree` 速查卡（5 条命令）

```bash
# 1) 创建草稿分支
git branch ai/draft-order-pricing

# 2) 给草稿分支开第二目录
git worktree add ../my-project-ai ai/draft-order-pricing

# 3) 查看当前所有 worktree
git worktree list

# 4) 对照主线与草稿差异
git diff main..ai/draft-order-pricing

# 5) 结束后清理
git worktree remove ../my-project-ai && git branch -D ai/draft-order-pricing
```

## 行动号召（CTA）

选你当前项目的一个核心用例，按本文流程跑一轮：  
先写规则与伪代码，再生成 AI 草稿，最后在 `main` 自主提交核心实现。
