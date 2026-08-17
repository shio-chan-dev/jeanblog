---
title: "从复合意图到可执行查询：面向实体的多源证据编排"
subtitle: "它不是图谱查询，而是连接自然语言、异构数据源与可追溯结果的语义查询计划"
date: 2026-08-17T12:00:00+08:00
categories: ["工程架构"]
tags: ["AI", "查询规划", "实体解析", "多源检索", "证据编排"]
summary: "当一个自然语言要求同时涉及多个主体、结构化事实、附件和原文时，可以先将复合意图编译为带主体的原子查询，再完成实体绑定、多源执行与按主体归并。知识图谱只是可选的主体解析工具，不是这套模式的本体。"
description: "介绍一种面向实体的多源证据编排模式：将复合意图转换为语义查询计划，经实体绑定和任务展开后调用不同数据源，最后按主体组装可追溯结果。"
keywords: ["语义查询规划", "多源查询", "实体绑定", "证据编排", "异构数据源", "知识图谱"]
readingTime: "约 11 分钟"
toc: true
---

> 这不是一种图谱查询方法，也不只是 RAG。它真正解决的是：如何把一句同时包含多个对象、多种信息形态和归属要求的自然语言请求，转换成一组可执行、可追溯、可重新归并的查询任务。

## 目标读者

- 正在设计智能检索、Agent 或企业信息系统的工程师
- 需要联合查询数据库、文件、文档和知识图谱的架构设计者
- 遇到“数据召回了，但无法确定属于谁”问题的开发者

## 真正的场景：供应商合规审查

假设用户提出这样的要求：

> 汇总审查范围内所有供应商的基本信息，附上每家供应商的经营许可证，并引用各自合同中的续期条款。

这不是一次简单搜索，而是一个复合查询：

1. “所有供应商”表示一个需要展开的主体集合。
2. “基本信息”是结构化事实。
3. “经营许可证”是必须归属于具体供应商的附件。
4. “各自合同中的续期条款”是需要保留原意的文档证据。
5. 最终结果必须重新按供应商组织，不能把甲公司的许可证放进乙公司的结果中。

数据通常分散在不同系统：

```text
供应商主数据       -> 数据库或业务 API
供应商与许可证关系 -> 关系表或知识图谱
许可证扫描件       -> 文件或对象存储
合同续期条款       -> 文档检索系统
```

如果直接做向量检索，可能找回很多相关文本，却无法保证供应商是否齐全、附件属于谁。

如果先让模型选择 `database`、`graph`、`document` 或 `hybrid`，它只是说出了数据在哪里，仍然没有形成可执行任务。

真正需要的是一层 **面向实体的多源证据查询编排**。

## 它为什么不等于图谱查询

图谱查询通常已经有相对明确的查询目标：

```text
查询所有 Supplier 节点
沿 HAS_LICENSE 关系找到 License
```

它解决的是图中的实体、关系和路径问题。

本文讨论的系统还要处理图外信息：

- 从供应商主数据服务获得事实。
- 从文件存储取得许可证扫描件。
- 从合同文档中取得续期条款原文。
- 将不同来源的结果重新归到同一个供应商名下。

因此，图谱只可能承担其中两个职责：

1. 把“所有供应商”“某个业务角色”等抽象主体解析为具体实体。
2. 提供实体之间的归属、参与和持有关系。

如果这些关系已经保存在关系型数据库或业务 API 中，系统完全可以不使用知识图谱。这套编排方式仍然成立。

一个简单判断是：

| 问题 | 更接近什么 |
| --- | --- |
| 已知节点和关系，查询图中的路径或邻居 | 图谱查询 |
| 从文本问题生成 Cypher 并查询一个图 | NL2Cypher |
| 从文档中召回相关片段并生成答案 | RAG |
| 将复合意图拆成实体级任务，联合多种数据源再归并 | 多源证据查询编排 |

## 核心模型：像编译器一样处理查询意图

这套链路可以类比为一个小型编译器：

```text
自然语言复合意图
        ↓
Semantic Planner
生成带主体的查询需求中间表示
        ↓
Entity Grounding
将抽象主体绑定到真实实体
        ↓
Task Lowering
将集合需求展开为实体级原子任务
        ↓
Source Executors
分别查询事实、附件和原文
        ↓
Entity Aggregation
按主体重新组织结果
```

它包含两次关键转换：

```text
用户想做什么 -> 系统需要哪些证据
抽象地说谁   -> 实际查询哪些对象
```

前者解决意图混合，后者解决实体指代和一对多展开。

## 第一层：生成与数据源无关的查询计划

Planner 只描述业务上需要什么，不决定数据来自哪里。

供应商合规请求可以转换成：

```json
{
  "requirements": [
    {
      "kind": "fact",
      "subject": {
        "type": "supplier",
        "scope": "all"
      },
      "instruction": "供应商基本信息"
    },
    {
      "kind": "material",
      "subject": {
        "type": "supplier",
        "scope": "all"
      },
      "instruction": "经营许可证"
    },
    {
      "kind": "document",
      "subject": {
        "type": "supplier",
        "scope": "all"
      },
      "instruction": "该供应商合同中的续期条款"
    }
  ]
}
```

这里使用三种证据需求：

| 类型 | 目标 | 典型结果 |
| --- | --- | --- |
| `fact` | 获得主体的结构化事实 | 名称、状态、属性、关系 |
| `material` | 获得需要核验或展示的附件 | 图片、证书、合同附件 |
| `document` | 获得必须忠实引用的原文 | 条款、原表、报告段落 |

每项 requirement 都携带自己的 subject。这样不会出现两个平行数组之间的隐式配对：

```text
subjects     = [供应商, 合同]
requirements = [许可证, 续期条款]
```

平行数组没有明确说明谁需要什么。把 subject 放进 requirement，数据结构本身就表达了绑定关系。

## 第二层：实体绑定与任务展开

Planner 输出的 `supplier, scope=all` 仍然不是具体查询。系统需要先解析审查范围，再得到真实供应商：

```text
supplier, scope=all
    -> supplier-001
    -> supplier-002
```

这个过程叫做 **Entity Grounding**。它可以由图谱、数据库、主数据服务或业务 API 完成。

随后，三项集合需求被展开为六个原子任务：

```text
fact(supplier-001, 基本信息)
fact(supplier-002, 基本信息)

material(supplier-001, 经营许可证)
material(supplier-002, 经营许可证)

document(supplier-001, 合同续期条款)
document(supplier-002, 合同续期条款)
```

这个一对多转换类似编译器中的 lowering：把高层表达降级成执行层能够直接处理的任务。

实体绑定不是新的证据查询链。它只是所有实体型任务共享的准备阶段。

## 第三层：不同执行器完成原子查询

任务展开之后，每个执行器只处理一种证据形态。

### Fact Executor

输入具体主体，向主数据服务、数据库或图谱查询结构化事实：

```text
supplier-001
    -> 企业名称
    -> 统一标识
    -> 注册状态
    -> 联系信息
```

执行器不需要理解完整用户问题，也不负责组织最终报告。

### Material Executor

输入主体和材料要求，查询该主体绑定的附件：

```text
supplier-001 + 经营许可证
    -> 候选附件
    -> 选择匹配材料
    -> 返回可访问资源及来源
```

主体先于材料解析，可以避免从一堆混合图片中重新猜测归属。

### Document Executor

输入主体和原文要求，查询与该主体关联的文档上下文：

```text
supplier-001 + 合同续期条款
    -> 将主体限定加入检索条件
    -> 定位对应合同
    -> 返回条款原文
```

有些 document 需求不属于特定主体，例如“引用统一采购政策”。这时 subject 可以为空，作为全局原文任务执行。

## 第四层：查询完成后按实体归并

执行结果不应该按照数据源返回给上层：

```text
database_results
file_results
document_results
```

调用方真正关心的是每个供应商获得了哪些证据：

```json
{
  "subject": {
    "id": "supplier-001",
    "type": "supplier"
  },
  "facts": ["..."],
  "materials": ["..."],
  "original_context": "...",
  "unresolved": []
}
```

因此，结果聚合应在各类查询完成后进行：

```text
执行原子任务
    ↓
保留 task -> subject 映射
    ↓
使用 subject_key 归并结果
    ↓
报告生成器消费实体级证据组
```

查询失败或无法归属的内容进入 `unresolved`。这不是附属日志，而是正式业务结果：它告诉调用方哪些要求尚未被证据覆盖。

## 最小可运行示例

下面的 Python 示例只实现“查询计划 -> 实体级任务”的转换，不依赖 LLM、图数据库或向量库。

```python
from dataclasses import dataclass
from typing import Literal


Kind = Literal["fact", "material", "document"]
Scope = Literal["one", "all"]


@dataclass(frozen=True)
class SubjectQuery:
    type: str
    scope: Scope


@dataclass(frozen=True)
class Requirement:
    kind: Kind
    subject: SubjectQuery | None
    instruction: str


@dataclass(frozen=True)
class Entity:
    id: str
    type: str


@dataclass(frozen=True)
class QueryTask:
    kind: Kind
    subject: Entity | None
    instruction: str


entities = [
    Entity("supplier-001", "supplier"),
    Entity("supplier-002", "supplier"),
]


def ground(subject: SubjectQuery) -> list[Entity]:
    matches = [entity for entity in entities if entity.type == subject.type]
    return matches if subject.scope == "all" else matches[:1]


def lower(
    requirements: list[Requirement],
) -> tuple[list[QueryTask], list[Requirement]]:
    tasks: list[QueryTask] = []
    unresolved: list[Requirement] = []

    for requirement in requirements:
        if requirement.subject is None:
            if requirement.kind == "document":
                tasks.append(
                    QueryTask("document", None, requirement.instruction)
                )
            else:
                unresolved.append(requirement)
            continue

        subjects = ground(requirement.subject)
        if not subjects:
            unresolved.append(requirement)
            continue

        tasks.extend(
            QueryTask(
                requirement.kind,
                subject,
                requirement.instruction,
            )
            for subject in subjects
        )

    return tasks, unresolved


all_suppliers = SubjectQuery("supplier", "all")
plan = [
    Requirement("fact", all_suppliers, "供应商基本信息"),
    Requirement("material", all_suppliers, "经营许可证"),
    Requirement("document", all_suppliers, "合同续期条款"),
]

tasks, unresolved = lower(plan)

for task in tasks:
    subject_id = task.subject.id if task.subject else "global"
    print(task.kind, subject_id, task.instruction)

assert len(tasks) == 6
assert unresolved == []
```

运行结果对应两个供应商的事实、材料和原文任务，共六项。把 `ground()` 替换成 SQL、图查询或业务 API，核心模型不会改变。

## 这套模式的真正不变量

无论底层使用什么技术，都应保持以下规则：

1. 一个复合意图可以拆成多个独立证据需求。
2. 每项需求明确说明它服务于哪个主体。
3. 集合主体必须展开为具体实体后再执行。
4. 多数据源结果通过主体标识重新关联。
5. 无法确认的归属必须显式返回，不能猜测。
6. 上层消费业务证据，不感知底层数据源路由。

这些规则才是设计的本体。知识图谱、向量库和关系数据库只是可替换实现。

## 图谱在什么情况下特别合适

虽然这不是图谱查询架构，但以下问题确实很适合交给知识图谱：

- “法定代表人”对应哪个 Person？
- 某个供应商持有哪些许可证？
- 某个人参与过哪些项目？
- 某份附件支撑哪个资质或合同？
- 一个主体与材料之间的关系在什么时间有效？

这些事实的核心语义存在于关系上，而不是某个孤立字段中。图谱可以显著降低实体绑定和归属查询的复杂度。

但图谱不擅长替代所有数据源：扫描件仍应由文件系统承载，长篇合同原文仍适合文档检索，运营状态仍可能来自业务数据库。

## 可以迁移到哪些领域

只要任务同时包含“多主体、多证据形态和结果归属”，这套模式就有价值。

| 场景 | 结构化事实 | 材料 | 原文 |
| --- | --- | --- | --- |
| 供应商合规 | 主体状态、许可证信息 | 许可证扫描件 | 合同条款 |
| 企业尽调 | 公司、股东、项目关系 | 营业执照、资质证书 | 审计报告、合同原文 |
| 保险理赔 | 保单、人员、事故事实 | 发票、照片、证明 | 病历或事故报告 |
| 合同审查 | 当事人、金额、期限 | 签章页、附件 | 条款和补充协议 |
| 科研证据整理 | 作者、机构、实验事实 | 图表、附录 | 论文方法和结论原文 |

这说明它更接近 **语义查询规划** 和 **联邦查询编排**，而不是某个具体数据库的查询技巧。

## 成本为什么可能下降

这层编排增加的是确定性数据转换，不一定增加模型调用。

成本下降通常来自：

1. 在执行前明确主体，避免召回大量无关对象。
2. 将 `scope=all` 交给数据系统展开，不让生成模型从混合文本中枚举。
3. 只把匹配后的材料和原文送给下游模型。
4. 把无法满足的需求作为结构化状态返回，避免模型反复尝试和猜测。

真正影响 Token 的是送入模型的上下文规模，而不是中间任务数量。十个精准查询可能比一次返回全部资料的宽泛查询更便宜。

## 常见错误

### 1. 把数据源名称当成业务计划

`graph`、`document` 和 `hybrid` 只说明如何检索，没有说明需要什么证据以及证据属于谁。

### 2. 让执行器重新理解完整意图

如果每个执行器都重新解释用户原话，多个分支可能得到不同的主体和范围。语义决定应该在 Planner 阶段完成一次。

### 3. 把 requirement 与 subject 分开存放

平行数组会重新引入配对歧义。需求应直接携带自己的主体。

### 4. 在实体绑定阶段提前组装最终结果

主体解析只负责生成任务。过早创建结果分组会把准备、执行和展示耦合在一起。

### 5. 把无法归属的材料强行绑定

没有足够来源信息时，正确输出是 `unresolved`，而不是看起来完整但可能错误的结果。

## 什么时候不值得使用

以下情况直接查询通常更简单：

- 只查询一个数据源。
- 用户已经提供明确实体 ID 和查询字段。
- 只需要找几段相关文本，不要求结果完整或按主体归属。
- 数据规模很小，不存在集合展开和材料错配。

判断标准可以压缩成一句话：

> 如果系统只需回答“什么内容相关”，普通检索通常足够；如果还要回答“需要什么、属于谁、是否齐全”，就需要查询计划和实体归并。

## 最佳实践

- 使用数据源无关的 requirement 作为中间表示。
- 让每项 requirement 直接携带 subject。
- 将实体绑定设计成独立的确定性准备阶段。
- 将集合需求展开成可并行执行的原子任务。
- 查询完成后再按 subject_key 归并结果。
- 将 `unresolved` 纳入正式返回契约。
- 分别测试规划正确性、实体覆盖率、材料归属和原文忠实度。

## 小结

这套思路的本质不是图谱查询，而是 **把复合自然语言意图编译成面向实体的多源证据任务**。

图谱可以帮助回答“谁与谁有关”，文档检索可以回答“原文写了什么”，文件存储可以提供“证明材料在哪里”，数据库可以返回“当前事实是什么”。查询计划负责把这些能力组织成一个可执行整体。

最值得保留的心智模型是：

```text
复合意图
-> 带主体的证据需求
-> 真实实体级任务
-> 多源执行
-> 按主体归并
```

一旦这层中间表示稳定，底层数据源可以演进，上层写作或分析流程也不必跟着重写。

## 参考与延伸阅读

- 编译器中的 Intermediate Representation 与 lowering
- Database Query Planning 与 Federated Query
- Entity Resolution 与 Entity Linking
- Knowledge Graph 中的关系建模
- Retrieval-Augmented Generation 中的证据组织
