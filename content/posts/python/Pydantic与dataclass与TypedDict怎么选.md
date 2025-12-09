---
title: "Pydantic vs dataclass vs TypedDict：谁负责什么，怎么组合？"
date: 2025-12-09
draft: false
---

### 标题

**Pydantic vs dataclass vs TypedDict：谁负责什么，怎么组合？**

---

### 副标题 / 摘要

承接《别让 Pydantic 占领你的整个项目》，这一篇用对比视角把 Pydantic、dataclass、TypedDict 的定位、取舍和组合方式讲清楚：**谁用于 API 校验、谁承载业务状态、谁只做类型提示**。

---

### 目标读者

* FastAPI / Pydantic 用户，想搞清楚“数据类”该放在哪一层
* 有 0–5 年经验、在做服务端建模的 Python 工程师
* 已读过前一篇分层文章，想进一步对比具体工具

---

### 背景 / 动机：为什么要区分三者？

在上一篇里，我们强调“Pydantic 应该停留在 API/外围”。很多同学随后会问：

* “那 Python 原生 dataclass 呢？和 Pydantic 有什么差？”
* “TypedDict 是不是又一个‘数据类’，要不要取代 dataclass？”
* “什么时候该用 Pydantic dataclasses，什么时候用标准库？”

不区分清楚，常见后果有：

* 用 TypedDict 写业务逻辑，测试时才发现它根本不做运行时校验；
* 用 Pydantic BaseModel 传来传去，导致 Domain 强绑定外部依赖；
* dataclass 和 Pydantic 混用，序列化和校验边界越来越模糊。

---

### 核心概念：一句话定位

* **Pydantic BaseModel**：运行时校验 + 类型转换 + JSON 友好；属于“对外/边界”。
* **dataclass（标准库）**：轻量数据载体，可承载业务方法；不做自动校验，属于“领域/内部”。
* **TypedDict**：仅提供静态类型提示，运行时就是普通 dict；属于“静态约束/外部协议”。

主要差异：

| 维度             | Pydantic BaseModel           | dataclass（stdlib）        | TypedDict                     |
| ---------------- | ---------------------------- | -------------------------- | ----------------------------- |
| 运行时校验       | ✅ 自动验证/转换             | ❌ 默认没有                | ❌ 完全没有                   |
| 序列化 JSON      | ✅ `model_dump`/`model_dump_json` | ⚠️ 需要手写/自定义        | ✅ 直接当 dict 用             |
| 依赖/重量        | 较重，依赖 Pydantic          | 轻量，纯标准库             | 最轻，纯类型标注             |
| 适合位置         | API DTO / 配置 / 外部数据    | Domain 实体 / 内部状态     | 第三方协议 / 配置静态约束    |
| 典型缺点         | 过度使用会侵入业务           | 需自带校验/转换            | 没有运行时保护，易遗漏字段   |

---

### 实践指南 / 选择步骤

1) **先画数据流**：从“外部输入 → 应用 → 存储/调用外部”三个方向，把边界找出来。  
2) **外部输入/输出（API、配置、Webhooks）**：用 Pydantic BaseModel，获取即时校验与错误信息。  
3) **内部业务状态**：用 dataclass 或普通类封装行为（方法）与状态；校验逻辑由业务方法/工厂函数负责。  
4) **外部协议但不需运行时校验**：用 TypedDict 给 dict 增强类型提示（如第三方 webhook payload）；若需要防御式校验，再包一层 Pydantic。  
5) **转换明确化**：用函数封装转换，例如 `req_to_domain(req: CreateOrderRequest) -> Order`，而不是在各处隐式组装。  
6) **必要时的折衷**：小型脚本/一次性任务可以只用 Pydantic；但一旦出现复用和演进需求，就落回分层模式。

---

### 可运行示例：三者在一个用例中的分工

```python
from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel, ValidationError, Field


# 1) 外部输入：HTTP 请求体 → Pydantic 负责校验与转换
class CreateOrderRequest(BaseModel):
    user_id: int
    sku: str
    quantity: int = Field(gt=0, default=1)


# 2) 内部业务状态：dataclass + 行为
@dataclass
class Order:
    user_id: int
    sku: str
    quantity: int
    status: str = "created"

    def total_price(self, unit_price: int) -> int:
        return self.quantity * unit_price

    def mark_paid(self):
        self.status = "paid"


# 3) 外部协议（第三方支付回调）：TypedDict 提供静态提示
class PaymentWebhook(TypedDict):
    order_id: int
    paid: bool
    gateway: Literal["stripe", "paypal"]


def create_order(raw_payload: dict, unit_price: int) -> Order:
    req = CreateOrderRequest.model_validate(raw_payload)  # runtime 校验
    order = Order(user_id=req.user_id, sku=req.sku, quantity=req.quantity)
    print("Order total:", order.total_price(unit_price))
    return order


def handle_webhook(payload: PaymentWebhook) -> str:
    # 这里只依赖 TypedDict 的键/值类型；需要防御时可再用 Pydantic 包一层
    if payload["paid"]:
        return f"{payload['gateway']} paid order {payload['order_id']}"
    return "payment failed"


if __name__ == "__main__":
    try:
        order = create_order({"user_id": 1, "sku": "ABC-123", "quantity": 2}, unit_price=199)
        print(order)
        print(handle_webhook({"order_id": 1, "paid": True, "gateway": "stripe"}))
    except ValidationError as e:
        print("Validation error:", e)
```

运行方式：

```bash
python demo.py
```

你会得到 Pydantic 的错误提示（如果参数错误），dataclass 的业务方法输出，以及 TypedDict 参与的回调处理。

---

### 解释与原理：为什么要这样分？

* **运行时安全 vs 静态约束**：Pydantic 提供即时反馈，适合边界；TypedDict 只在类型检查器里生效，运行时不阻止坏数据。  
* **行为聚合**：dataclass 能挂方法（业务规则、状态变更），保持“对象 + 行为”的一致性；TypedDict 只是结构声明，不能挂行为。  
* **依赖方向**：让“内层”只依赖标准库（dataclass），把第三方依赖挡在外层（Pydantic）。这样测试、迁移框架都更从容。  
* **性能与开销**：Pydantic 解析会有开销，高吞吐内部循环不宜使用；dataclass 纯 Python，TypedDict 近乎零开销。

---

### 常见问题与注意事项

1) **Pydantic dataclasses 能替代 stdlib dataclass 吗？**  
   小项目可以，但它会引入 Pydantic 依赖与校验开销，违背“内核只依赖标准库”的目标。

2) **TypedDict 会报缺字段的错误吗？**  
   不会。mypy/pyright 能提示，运行时依旧是普通 dict；若想要运行时保护，用 Pydantic 校验再转成 TypedDict。

3) **Domain 校验放哪？**  
   把不变式写在 dataclass 的工厂函数/方法里，例如在 `__post_init__` 或自定义构造里校验状态，而不是依赖 Pydantic。

4) **什么时候可以“偷懒”只用 Pydantic？**  
   Demo、一次性脚本或非常薄的 CRUD 服务可以；但一旦需要复用领域逻辑或拆分模块，尽早抽出 dataclass。

5) **与 ORM 怎么配？**  
   ORM 层（SQLAlchemy/SQLModel）可以把查询结果映射成 dataclass。Pydantic 用于 API 输入输出，不要让 ORM 模型上浮到业务层。

---

### 最佳实践与建议

1. 边界输入输出 → Pydantic BaseModel（或 Settings）做校验与转换。  
2. 领域实体 → dataclass/普通类，写入业务方法与不变式。  
3. 外部协议/第三方 payload → TypedDict 约束静态类型，必要时再包一层 Pydantic。  
4. 永远显式转换，避免“随处 dict 拼装”。  
5. 测试 Domain 时不引入 Pydantic/ORM；测试边界时用 Pydantic 提供的错误信息。  
6. 如果性能敏感，避免在热路径里频繁创建 Pydantic 模型。

---

### 小结 / 结论

* **Pydantic**：跑在边界，负责“把数据弄干净”。  
* **dataclass**：守在业务核心，承载状态与行为。  
* **TypedDict**：给 dict 加静态护栏，别指望它在运行时救你。  
把三者放在对的位置，转换写清楚，你就能既享受校验的便利，又保持业务内核的纯粹。

---

### 参考与延伸阅读（按关键词搜索）

* “Pydantic BaseModel validation vs dataclasses”  
* “Python dataclass best practices domain model”  
* “TypedDict runtime vs static type checking”  
* “Clean Architecture Python Pydantic SQLAlchemy”  
* “FastAPI DTO vs domain model”  

---

### 元信息

* **预计阅读时长**：10–14 分钟  
* **标签（Tags）**：Pydantic, dataclass, TypedDict, FastAPI, 分层架构, Python 类型系统  
* **SEO 关键词（Keywords）**：Pydantic dataclass TypedDict 区别, Python 数据建模选择, FastAPI DTO 校验, Domain 模型 dataclass, TypedDict 用法, Pydantic 校验示例  
* **元描述（Meta Description）**：本篇承接“别让 Pydantic 占领你的整个项目”，对比 Pydantic、dataclass、TypedDict 的职责与适用场景，提供可运行示例和选择指南，帮助你在 API 校验、领域建模和外部协议之间正确落位。

---

### 行动号召（CTA）

* 🛠 **动手重构**：挑一个核心实体，把 API 层的 Pydantic 模型转换为 dataclass 领域对象，再写一层转换函数。  
* 🧪 **开个类型检查**：给第三方回调 payload 写一个 TypedDict，并跑一次 mypy/pyright，体验静态提示带来的安全感。  
* 📥 **订阅/收藏**：如果想看更多分层与建模的实战案例，订阅后续更新或把这篇加入书签，方便对照改造你的项目。
