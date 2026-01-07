---
title: "算法与业务的关系：把不确定性变成可交付（以 LLM 事实抽取为例）"
subtitle: "别再用工程代码调黑箱：用 Notebook 让算法收敛，再让业务稳定交付"
date: 2026-01-07
draft: false

description: "用投标写作系统为例，解释算法与业务的边界：算法负责把模糊世界压缩成结构化中间态，业务负责在中间态稳定后编排流程。附：JupyterLab 验证清单、最小可运行示例与常见坑。"
tags: ["工程实践", "LLM", "RAG", "架构", "NLP"]
categories: ["工程架构"]
keywords: ["算法与业务", "JupyterLab", "LLM抽取", "事实抽取", "中间态", "RAG", "投标写作", "系统设计", "Prompt工程"]
readingTime: "10 min"
toc: true
---

## 摘要

很多团队做 AI 应用时会陷入一种痛苦：**业务逻辑写了一堆，却始终看不到“算法到底抽出了什么”**。根因往往不是代码能力，而是**把算法阶段的探索，过早塞进业务工程**。本文用“投标写作工具里的事实抽取（FactMention）”作为例子，讲清楚算法与业务的边界、如何用 JupyterLab 快速验证、以及何时进入工程化。

---

## 目标读者

- 正在做 LLM/RAG/信息抽取的工程师（初级到中级）
- 负责 AI 产品落地的技术负责人 / 架构师
- 经常在“写了半天流程，结果不知道抽取效果如何”的同学

---

## 背景与动机：为什么这个问题重要？

在传统系统里，大家习惯把“算法”理解为一个函数或模型文件，把“业务”理解为接口与流程。但在 LLM 时代，这个界限变得更模糊：

- 算法不仅是模型，还包括 **prompt、schema、抽取策略、规则归一、置信度与去重策略**
- 算法输出往往是 **不确定、需要人类直觉评估** 的
- 如果你把这些“不确定”的东西直接嵌进业务链路（router/service/db/cache），你会遇到：
  - 调试成本爆炸：只看到最后 response，不知道中间发生了什么
  - 逻辑迭代极慢：改一行抽取策略，要跑完整流程
  - 团队协作困难：大家在黑箱里争论“到底准不准”

因此，需要一个更清晰的边界：**算法负责收敛中间态，业务负责稳定交付。**

---

## 核心概念：算法与业务到底分别是什么？

### 1）算法（Algorithm）的工程定义

> **算法负责把“模糊世界”压缩成“可用的结构化中间态”。**

关键词：不确定性、探索、需要“看结果”、需要收敛。

在投标写作场景中，算法阶段的问题长这样：

- LLM 抽取出来的 `payload` 字段应该有哪些？
- `norm_key` 怎么设计才代表“同一事实”？
- 同一人多条命中（mentions）要不要合并？
- `confidence` 到底有没有意义？怎么校准？

这些问题的共同特点是：**你必须看中间结果才能判断对不对**。

---

### 2）业务（Business）的工程定义

> **业务负责在中间态稳定之后，把事情编排起来：什么时候取什么数据、走哪条链路、如何返回给用户。**

关键词：确定性、可维护、可测试、可复用。

在投标写作场景中，业务阶段的问题长这样：

- `retrive_type = personnel` 时走事实检索，否则走原文档检索
- 接口响应结构固定，前端按协议渲染
- 存储从 MemoryTable 换成 DB，不影响上层调用

这些问题的共同特点是：**输入输出清晰，错误是边界情况，而不是“我也不知道会不会抽出来”。**

---

## 一条“贴墙上”的分界线

> **凡是你还说不清“中间结果长什么样”的阶段 → 用 JupyterLab。**  
> **凡是你能画出输入/输出 JSON 形态并写出测试用例的阶段 → 进工程。**

---

## 实践指南：什么时候用 JupyterLab，什么时候用工程代码？

### 阶段 1：事实抽取建模期（强制用 JupyterLab）

**适用信号：**
- 你还在调整 prompt / schema / 规则归一
- 你关心“到底抽出来了什么”

**目标：**
- 让 `FactMention` 的结构和质量收敛到“你敢在写作里用”的程度

**Notebook 里应该做：**
- 单文件抽取 → 打印 mentions（payload + evidence.span）
- 多文件抽取 → 统计分布（同一 norm_key 出现次数）

---

### 阶段 2：可用性验证期（Notebook 为主 + 少量工程壳）

**适用信号：**
- 抽取稳定了，但你在验证“怎么用更合理”
- 需要做筛选、排序、组合

**目标：**
- 验证写作节点的 needs（例如 personnel/project/qualification）能否正确路由到事实检索

---

### 阶段 3：接口稳定期（工程为主，Notebook 退居实验室）

**适用信号：**
- 输入输出结构稳定
- 你能明确写出：如何回归测试抽取效果

**目标：**
- 把可用策略固化为服务、缓存、权限、审计、并发处理

---

## 可运行示例：最小事实抽取中间态（Python）

> 下面示例演示“为什么要用中间态”，并给出一个极简 pipeline：  
> 文本 → LLM（模拟）→ FactMention → 规则归一 → 输出

> 你可以把它复制到 Notebook 里跑（把 mock_llm_extract 换成你的 LLM 调用）。

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import hashlib

@dataclass
class Evidence:
    file_id: str
    doc_id: str
    node_id: str
    page: int
    span: str
    confidence: float

@dataclass
class FactMention:
    mention_id: str
    fact_type: str            # personnel / project / qualification ...
    payload: Dict[str, Any]   # e.g., {"name": "...", "role": "..."}
    evidence: Evidence
    norm_key: str

def normalize_personnel(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 规则归一示例：去空格、统一角色名等（按你业务扩展）
    p = dict(payload)
    if "name" in p and isinstance(p["name"], str):
        p["name"] = p["name"].strip()
    if "role" in p and isinstance(p["role"], str):
        p["role"] = p["role"].strip()
    return p

def make_norm_key(payload: Dict[str, Any]) -> str:
    # 一个简单 norm_key：name|role（可扩展：证书/职称/身份证明等）
    base = f'{payload.get("name","")}|{payload.get("role","")}'
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

def mock_llm_extract(text: str) -> List[Dict[str, Any]]:
    # 模拟 LLM 结构化输出（真实情况是 LLM JSON + Pydantic 校验）
    # 这里只演示形态
    if "张三" in text and "项目经理" in text:
        return [{
            "fact_type": "personnel",
            "payload": {"name": "张三", "role": "项目经理", "title": "高级工程师", "certificates": ["PMP"], "years": 10},
            "evidence": {"page": 12, "span": "拟任项目经理张三，10年经验，持PMP证书", "confidence": 0.75}
        }]
    return []

def extract_mentions(file_id: str, doc_id: str, node_id: str, text: str) -> List[FactMention]:
    raw = mock_llm_extract(text)
    mentions: List[FactMention] = []
    for i, item in enumerate(raw):
        if item["fact_type"] == "personnel":
            payload = normalize_personnel(item["payload"])
        else:
            payload = item["payload"]

        norm_key = make_norm_key(payload)
        ev = Evidence(
            file_id=file_id,
            doc_id=doc_id,
            node_id=node_id,
            page=item["evidence"]["page"],
            span=item["evidence"]["span"],
            confidence=item["evidence"]["confidence"],
        )
        mentions.append(FactMention(
            mention_id=f"{doc_id}-{node_id}-{i}",
            fact_type=item["fact_type"],
            payload=payload,
            evidence=ev,
            norm_key=norm_key,
        ))
    return mentions

# demo
text = "……拟任项目经理张三，10年经验，持PMP证书……"
mentions = extract_mentions("file-123", "doc-uuid-001", "node-uuid-aaa", text)
mentions
```

---

## 解释与原理：为什么“FactMention + Evidence”比“原文 chunk”更适合业务写作？

在写作阶段你真正需要的是：

1. **事实（payload）**：可复用、可组合、可筛选
2. **证据（evidence）**：可追溯、可审计、可 debug

如果你只返回原文 chunk：

* 它可能混着多条事实（同页多条、同段多人）
* 你难以在写作时做“精确选择与组合”
* 当用户质疑“你怎么填的？”时，没有可解释来源

而 `FactMention` 让你能做到：

* **同页多条事实** → 多条 mentions
* **同一事实多处证据** → 通过相同 `norm_key` 在生成时“软聚合”
* **写作强约束**：prompt 可要求“只能引用 facts，不得凭空编造”

---

## 替代方案与取舍：要不要做向量化？

**V1 不必须上向量库**，因为 personnel/project/qualification 多数是结构化筛选（角色、证书、年限）。
向量化更适合：

* 用户提出开放式语义问题：“找最像这个项目的案例”
* facts 规模很大，需要语义召回
* 想做“描述文本 → 找匹配事实”的模糊查询

建议路线：

* V1：不做向量库，先把事实抽取与可解释链路跑通
* V2：对 **FactRecord** 或 facts 集合单独建向量索引（不要挂 file_tree）

---

## 常见问题与注意事项（你大概率会踩）

### 1）为什么我“写了半天业务逻辑”但看不到抽取结果？

因为你把“算法探索”藏进了 router/service 之后。
做法：先在 Notebook 把 `mentions` 打印出来再进工程。

### 2）mentions 为空，到底是没抽到还是被过滤了？

要把 pipeline 拆成可观察步骤：

* LLM raw 输出
* Pydantic 校验失败原因
* 规则归一前后对比
* 最终 mentions 数量

### 3）是否应该把事实挂到 file_tree？

不建议挂事实本体。
可以挂引用（fact_id / mention_id）或摘要，避免 JSON 膨胀和并发冲突。

### 4）内存表进程重启会清空怎么办？

V1 正常。先验证流程。
当抽取质量和接口稳定后再换持久化（DB / KV / 向量库）。

---

## 最佳实践建议（经验总结）

* **先让中间态“可见”**：Notebook 里可视化 mentions（表格、统计、抽样）
* **先做软聚合再做硬合并**：V1 生成时按 norm_key 分组，不急着引入 FactRecord
* **事实与证据分离**：payload 是事实；evidence 是定位与审计
* **业务只做路由**：`retrive_type` 决定走 facts 还是文档 chunk，避免把抽取细节散落在业务层
* **用 schema 管 LLM**：结构化 JSON 输出 + 校验 + 归一，比“随便生成一段话”稳定得多

---

## 小结

* **算法**的职责：把不确定的文本世界，压缩成稳定的结构化中间态（FactMention）
* **业务**的职责：在中间态稳定之后，编排流程并交付（retrieve_type 路由、接口、存储）
* **JupyterLab**是算法阶段的“观察窗”，让你快速看到抽取结果并收敛；工程化是稳定交付的“流水线”

下一步建议：
选一个真实投标文件，在 Notebook 里跑通 **单文件抽取 → mentions 可视化 → norm_key 分组统计**。当你能稳定回答“抽到了什么、为什么可信、怎么选用”时，再把它搬进服务层。

---

## 参考与延伸阅读

* Pydantic 文档：用于结构化校验与错误可解释（建议你在抽取中强制用 schema）
* RAG / 信息抽取工程实践：建议关注“可解释性”“可审计性”“中间态设计”
* Notebook 驱动开发（NDD）：用 Notebook 先收敛数据形态，再工程化

（如果你需要，我也可以按你的项目结构补一份“Notebook 骨架清单”和“进入工程的验收标准”。）

---

## 行动号召（CTA）

如果你正在做 LLM 抽取 / 投标写作 / 事实库复用系统：

1. 先在 Notebook 里把 `FactMention` 的样子“看清楚”
2. 再把 `retrive_type` 路由固化进工程
3. 最后才考虑 FactRecord 聚合与向量索引

欢迎把你现在的 `FactMention` 样例贴出来（脱敏即可），我可以帮你做一份“字段收敛与归一策略”的建议清单。
