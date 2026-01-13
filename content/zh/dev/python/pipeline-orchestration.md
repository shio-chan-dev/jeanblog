---
title: "任务编排为什么要放后端：让流程可控、可变、可回放"
subtitle: "前端只负责展示与输入，后端用 Pipeline 统一顺序与状态"
date: 2026-01-13
draft: false
description: "用通俗语言解释“后端任务编排”的必要性：把流程顺序从前端移到后端，用配置驱动步骤与等待点。给出最小可运行示例、落地步骤与常见坑。"
tags: ["工程实践", "架构", "Workflow", "LLM", "系统设计"]
categories: ["工程架构"]
keywords: ["任务编排", "Pipeline", "后端编排", "流程引擎", "状态机", "可回放", "可重试", "系统设计"]
readingTime: "10 min"
toc: true
---

# 任务编排为什么要放后端：让流程可控、可变、可回放

## 副标题 / 摘要

在多步骤、可中断、可回放的业务流程中，把“流程顺序与状态机”放在后端，是系统长期可演进的关键。本文从真实工程痛点出发，解释为什么前端不应硬编码流程顺序，并给出一套可落地的后端 Pipeline 编排思路与最小实现。

---

## 目标读者

* 正在设计**多步骤流程 / 向导式产品**的后端工程师
* 需要支撑 **Web / App / Admin 多端一致流程**的技术负责人
* 在 **AI / LLM 产品**中处理“模型自动 + 人工确认”混合流程的团队

---

## 背景 / 动机：问题通常是怎么爆出来的？

很多系统一开始都很简单：

> 前端：第 1 步 → 第 2 步 → 第 3 步
> 后端：校验 + 存数据

但随着业务演进，以下需求几乎一定会出现：

* 步骤 **变多**：从 3 步变成 10+ 步
* 步骤 **可选**：根据条件跳过 / 插入新步骤
* 步骤 **可中断**：需要用户确认、补充信息、人工审核
* 步骤 **可重试 / 可回放**：失败后从中间继续，而不是全部重来
* 步骤 **多端一致**：Web / App / 内部工具共享同一流程

如果此时流程顺序仍然写在前端：

* 每次流程变更 = 多端发版
* 出问题时无法准确回答：**现在卡在哪一步？**
* 想加监控、审计、回放，发现无从下手

**根因只有一个**：

> 流程是一等公民，却被当成了前端行为脚本。

---

## 核心观点（一句话版）

> **前端负责“展示与输入”，后端负责“顺序、状态与推进规则”。**

流程不应该存在于前端代码中，而应该存在于后端的：

* 配置（Pipeline / Workflow 定义）
* 状态机（当前在哪一步，是否可推进）
* 执行记录（每一步做了什么，产出了什么）

---

## 核心概念拆解（工程视角）

### 1️⃣ Task / Flow Instance（流程实例）

* 每次用户触发一个流程，都会生成一个 **Task ID**
* Task ID 是日志、监控、回放、审计的核心索引

> 一切问题都应该能回答：**“这个 Task 现在处在哪一步？”**

---

### 2️⃣ Pipeline / Workflow Definition（流程定义）

Pipeline 是**纯描述性配置**，而不是代码流程：

* 有哪些步骤（Steps）
* 步骤之间的依赖关系
* 哪些步骤是自动的，哪些需要用户参与
* 条件分支与可选路径

它的本质类似：

* 有限状态机（FSM）
* 或 DAG（有向无环图）

---

### 3️⃣ Step（步骤）

一个 Step 是最小可管理单元，通常具备：

* 输入（来自用户或上一步产物）
* 执行逻辑（自动 or 等待）
* 输出（Artifact）

典型分类：

* **AUTO**：后端可自动执行（计算、调用服务、跑模型）
* **WAIT_USER**：必须等前端提交输入才能继续

---

### 4️⃣ Artifact（步骤产物）

每一步都应该有“可记录的结果”，例如：

* 结构化 JSON
* 文件路径 / 对象存储 key
* LLM 推理结果

Artifact 的价值在于：

* 支持失败回放
* 支持跳过已完成步骤
* 支持审计与问题排查

---

## 一个真实场景示例（AI 产品）

以 **“上传文档 → AI 解析 → 人工确认 → 再处理”** 为例：

1. 用户上传文档（AUTO）
2. LLM 自动生成目录（AUTO）
3. 用户确认 / 修改目录（WAIT_USER）
4. 后端按最终目录拆分文档（AUTO）
5. 生成结构化数据 / 向量（AUTO）

如果流程写在前端：

* 目录确认步骤一改，所有端都要改
* 无法优雅支持“跳过确认”“重新确认”

如果流程在后端：

* 前端只关心：**现在是不是要我确认？**
* 后端随时可调整：是否强制确认、是否插入新步骤

---

## 后端编排的最小接口设计

前端真正需要的接口，其实非常少：

### 1️⃣ 查询当前流程状态

```json
GET /tasks/{task_id}

{
  "status": "WAITING_USER",
  "current_step": "directory_confirm",
  "required_input": {
    "type": "text",
    "schema": { "directory": "string" }
  }
}
```

### 2️⃣ 提交用户输入并推进流程

```json
POST /tasks/{task_id}/advance

{
  "step": "directory_confirm",
  "input": {
    "directory": "..."
  }
}
```

前端逻辑可以被极度简化为：

> 根据 `current_step` 渲染 UI，提交后刷新状态

---

## 可运行示例（概念级）

> 以下示例刻意简化，用于理解思想，而非生产级实现。

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Step:
    id: str
    type: str  # auto | wait_user
    depends_on: List[str]

PIPELINE = [
    Step("upload", "auto", []),
    Step("abstract", "auto", ["upload"]),
    Step("directory_confirm", "wait_user", ["abstract"]),
    Step("directory_parse", "auto", ["directory_confirm"]),
]

def can_run(step, done):
    return all(done.get(d) for d in step.depends_on)

def run(user_input=None):
    done = {}
    for step in PIPELINE:
        if not can_run(step, done):
            break
        if step.type == "wait_user" and not user_input:
            return {"status": "WAITING_USER", "step": step.id}
        done[step.id] = True
    return {"status": "COMPLETED", "done": list(done)}
```

---

## 为什么这套模式“长期更便宜”？

### 从工程成本看

| 维度    | 前端编排  | 后端编排 |
| ----- | ----- | ---- |
| 流程变更  | 多端修改  | 改配置  |
| 失败恢复  | 几乎不可行 | 天然支持 |
| 监控审计  | 分散    | 集中   |
| 多端一致性 | 难     | 易    |

---

## 常见坑与注意事项（血泪版）

* ❌ **步骤无幂等性**：一重试就写脏数据
* ❌ **状态只存在内存**：服务重启即丢流程
* ❌ **用户输入无 schema**：后期无法演进
* ❌ **流程无版本**：老任务跑新逻辑直接炸

---

## 最佳实践总结

* 流程 = 配置 + 状态，而不是前端代码
* 每一步都要可重试、可记录、可回放
* 前端永远不要“猜下一步”
* 先做线性 Pipeline，再进化到 DAG

---

## 小结 / 结论

后端任务编排不是为了“技术优雅”，而是为了：

> **让复杂流程在时间维度上依然可控。**

当流程可以被记录、暂停、回放、演进，你的系统才真正具备规模化与长期演进能力。

---

## 行动号召（CTA）

选一个你们**最常改、最容易出问题的流程**：

* 把顺序从前端删掉
* 用一个最小 Pipeline 描述它
* 让前端只渲染“当前步骤”

你会很快意识到：

> 流程一旦回到后端，系统就安静了很多。

