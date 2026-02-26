---
title: "先定不变量与契约，再写实现：Evans/Fowler 实战法"
date: 2026-02-11T07:53:37+08:00
draft: false
description: "解释“先定不变量/契约，再写实现”到底在工程上多了什么，并给出可执行的落地模板。"
tags: ["DDD", "不变量", "契约设计", "软件设计", "工程实践"]
categories: ["工程实践"]
keywords: ["不变量", "契约", "Evans", "Fowler", "Design by Contract"]
---

## 标题

先定不变量与契约，再写实现：Evans/Fowler 实战法

## 副标题 / 摘要

很多人理解“先定不变量与契约”时，会觉得只是“多写几行校验”。这篇文章给出更精确的答案：它的本质是固定责任归属，让调用方可以依赖行为语义，而不是猜测实现细节。

## 目标读者

- 正在做业务系统设计、代码评审的工程师
- 觉得“代码能跑，但改需求总出坑”的团队
- 想把 DDD/契约思想落到日常开发的人

## 背景 / 动机

常见开发顺序是“先把功能跑通，再补规则”。短期看速度快，长期会出现三个问题：

- 业务规则散落在多个 service/controller 里
- 调用方只能通过读实现猜行为
- 改一个需求会牵动大量分支判断

Evans/Fowler 这一脉的核心不是“写得更学术”，而是先明确系统必须成立的事实，再让实现为这些事实服务。

## 核心概念

- **不变量（Invariant）**：无论任何路径，始终为真的业务规则。  
  例如：已支付订单不能再次支付。
- **契约（Contract）**：对外可依赖的行为承诺，至少包含前置条件、后置条件、失败语义。  
  例如：`cancel(order)` 只接受可取消状态，成功后状态必须是 `CANCELLED`，否则抛明确异常。
- **接口 vs 契约**：接口是签名，契约是语义保证。  
  同一个函数签名，可以有强契约，也可以完全没有契约。

### 契约分层（建议团队统一术语）

前面的 `cancel(order)` 示例主要覆盖了**行为契约**与**失败契约**。  
在真实项目里，建议把契约至少拆成下面 6 类，一起设计：

- **数据契约**：输入/输出的数据形状、类型、取值范围、单位、精度、是否可空。  
  例：金额必须 `> 0`，币种必须是 `ISO 4217`，时间必须是 UTC。
- **状态契约**：状态机允许哪些迁移，不允许哪些迁移。  
  例：订单只能 `CREATED -> PAID -> SHIPPED`，不能 `SHIPPED -> CREATED`。
- **不变式契约**：跨方法、跨状态始终成立的事实。  
  例：订单总额 = 明细金额之和 + 运费 - 优惠；库存不可为负。
- **行为契约**：调用成功时，调用方可以依赖什么结果与语义。  
  例：`reserve_stock()` 成功后，一定返回预留记录 ID，且库存已被占用。
- **失败契约**：违约/异常时返回什么错误、错误是否可重试、是否有副作用残留。  
  例：重复请求返回 `409`；超时返回 `503` 且标记 `retryable=true`。
- **副作用契约**：方法会修改哪些外部状态（DB、缓存、消息、文件），顺序如何，失败如何补偿。  
  例：先写 DB 再写 outbox；缓存删除失败不影响主事务提交。

## 实践指南 / 步骤

1. **先写目的，不写实现**  
   明确本次功能要改变什么业务结果。
2. **列不变量清单**  
   逐条写出“绝对不能被破坏”的规则。
3. **定义契约**  
   为核心行为定义前置条件、后置条件、失败语义，并补齐数据/状态/副作用契约。
4. **再落实现**  
   数据库、框架、缓存、消息等实现细节后置。
5. **用测试锁契约**  
   测试验证的是契约，不是某一版实现细节。

## 可运行示例

### 示例 1：无契约（可运行，但语义模糊）

```python
class Order:
    def __init__(self, status):
        self.status = status


def cancel(order: Order) -> Order:
    if order.status != "CREATED":
        return order
    order.status = "CANCELLED"
    return order


if __name__ == "__main__":
    order = Order("PAID")
    after = cancel(order)
    print(after.status)
```

问题：失败是“静默返回”，调用方必须自己猜“这次到底算成功还是失败”。

### 示例 2：有契约（调用方可依赖）

```python
class CannotCancelOrder(Exception):
    pass


class Order:
    def __init__(self, status):
        self.status = status

    def cancel(self):
        if self.status != "CREATED":
            raise CannotCancelOrder(f"invalid status={self.status}")
        self.status = "CANCELLED"
        return self


if __name__ == "__main__":
    order = Order("CREATED")
    order.cancel()
    print(order.status)
```

这里的契约是：

- 前置条件：状态必须是 `CREATED`
- 后置条件：成功后状态一定是 `CANCELLED`
- 失败语义：违约时抛 `CannotCancelOrder`

这个例子主要体现的是**行为契约 + 失败契约**。

### 示例 2.1：把同一个 `cancel()` 行为拆成 6 类契约

| 契约类型 | `cancel(order)` 的例子 |
| --- | --- |
| 数据契约 | `order.id` 非空；`status` 必须是合法枚举值；取消原因长度 <= 200 |
| 状态契约 | 仅允许 `CREATED` / `PAID_PENDING` 进入取消；`SHIPPED` 不可取消 |
| 不变式契约 | 已取消订单不能再次支付；取消后订单总额不变（只是状态变化） |
| 行为契约 | 成功调用后返回的订单状态一定为 `CANCELLED` |
| 失败契约 | 不可取消时抛 `CannotCancelOrder`，而不是静默返回原对象 |
| 副作用契约 | 成功取消后写审计日志；若有库存预留则释放库存；失败时不写取消日志 |

### 示例 3：支付创建（数据契约 + 失败契约）

下面这个例子重点展示：不是“校验字段”而已，而是把失败语义也定清楚。

```python
from dataclasses import dataclass
from decimal import Decimal


class ContractViolation(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass
class CreatePaymentCommand:
    order_id: str
    amount: Decimal
    currency: str
    request_id: str


def create_payment(cmd: CreatePaymentCommand) -> dict:
    # 数据契约
    if not cmd.order_id:
        raise ContractViolation("INVALID_ORDER_ID", "order_id is required")
    if cmd.amount <= Decimal("0"):
        raise ContractViolation("INVALID_AMOUNT", "amount must be > 0")
    if cmd.currency not in {"CNY", "USD"}:
        raise ContractViolation("INVALID_CURRENCY", "unsupported currency")
    if not cmd.request_id:
        raise ContractViolation("INVALID_REQUEST_ID", "request_id is required")

    # 行为契约（示例化返回）
    return {
        "payment_id": "pay_001",
        "order_id": cmd.order_id,
        "status": "PENDING",
    }


if __name__ == "__main__":
    cmd = CreatePaymentCommand("order_1", Decimal("99.90"), "CNY", "req-123")
    print(create_payment(cmd))
```

这个例子可写出的契约包括：

- 数据契约：金额必须大于 0、币种必须受支持、`request_id` 必填
- 行为契约：成功后一定返回 `payment_id` 且状态为 `PENDING`
- 失败契约：输入不合法抛 `ContractViolation`（可被 API 层稳定映射）

### 示例 4：库存预留（状态契约 + 不变式契约）

```python
class InsufficientStock(Exception):
    pass


class InvalidSkuState(Exception):
    pass


class InventoryItem:
    def __init__(self, sku: str, available: int, status: str = "ACTIVE"):
        self.sku = sku
        self.available = available
        self.reserved = 0
        self.status = status

    def reserve(self, qty: int):
        # 数据契约
        if qty <= 0:
            raise ValueError("qty must be > 0")
        # 状态契约
        if self.status != "ACTIVE":
            raise InvalidSkuState(f"sku {self.sku} is not active")
        # 不变式契约（库存不可负）
        if self.available < qty:
            raise InsufficientStock(f"available={self.available}, qty={qty}")

        self.available -= qty
        self.reserved += qty
        # 后置条件 + 不变式
        assert self.available >= 0
        assert self.reserved >= 0
        return self


if __name__ == "__main__":
    item = InventoryItem("sku-1", 10)
    item.reserve(3)
    print(item.available, item.reserved)  # 7 3
```

这个例子里最关键的不只是“能不能 reserve”，而是你提前定义了：

- 状态契约：只有 `ACTIVE` 才允许预留
- 不变式契约：`available >= 0`
- 失败契约：库存不足抛明确异常，而不是返回 `False`

### 示例 5：副作用契约（事务主路径 vs 非关键副作用）

副作用契约最容易缺失，但工程影响最大。下面用一个简化示例表达“哪些副作用必须成功，哪些可以降级”。  
（这里不引入真实数据库，只模拟顺序和失败语义）

```python
class AuditLogError(Exception):
    pass


class OrderService:
    def __init__(self):
        self.db = {}
        self.audit_logs = []

    def _write_audit(self, message: str):
        # 模拟偶发失败
        raise AuditLogError("audit service unavailable")

    def cancel_order(self, order_id: str) -> dict:
        # 副作用契约（主路径）
        # 1) 订单状态更新必须成功，否则整体失败
        self.db[order_id] = "CANCELLED"

        # 副作用契约（非关键路径）
        # 2) 审计日志失败不回滚主事务，但必须记录告警/重试任务（此处用字段模拟）
        audit_pending = False
        try:
            self._write_audit(f"cancel {order_id}")
        except AuditLogError:
            audit_pending = True

        return {"order_id": order_id, "status": "CANCELLED", "audit_pending": audit_pending}


if __name__ == "__main__":
    svc = OrderService()
    print(svc.cancel_order("o-1"))
```

这里的重点契约不是代码技巧，而是你提前说清楚：

- 主副作用：订单状态更新成功才算成功
- 次副作用：审计日志失败不影响主事务结果
- 失败语义：返回 `audit_pending=True`（或写重试任务），而不是悄悄吞掉

### 示例 6：常见契约例子速查表（跨模块）

| 场景 | 契约类型 | 示例 |
| --- | --- | --- |
| HTTP API `POST /orders` | 数据契约 | `user_id` 必填；`items` 非空；金额字段精度固定到分 |
| HTTP API `POST /orders` | 失败契约 | 参数错误 `400`；重复请求 `409`；下游超时 `503` 且可重试 |
| 订单状态迁移 | 状态契约 | `PAID` 后才能 `SHIP`; `CANCELLED` 后不能 `PAY` |
| 账户扣款 | 不变式契约 | 余额不得为负；记账借贷和必须相等 |
| 缓存更新 | 副作用契约 | DB 提交成功后删除缓存；删缓存失败进入重试队列 |
| 幂等接口 | 行为契约 | 相同 `request_id` 重试返回同一业务结果，而不是重复创建 |

## 解释与原理

“先定不变量/契约，再写实现”并不等于“偏爱 OOP”。  
它真正解决的是责任分配：

- 没有契约时：调用方承担判断责任（读实现、猜结果、补防御）
- 有契约时：被调用方承担规则责任（成功保证、失败明确）

所以差别不在“有没有 class”，而在“调用方是否能闭眼依赖该行为”。

进一步说，在真实系统里，最容易出事故的往往不是“行为契约没写”，而是：

- **数据契约含糊**（金额单位、时区、可空性不清）
- **状态契约缺失**（状态迁移随处可改）
- **副作用契约模糊**（到底要不要回滚、要不要重试没人说清）

这也是为什么我建议把契约拆层，而不是只写“前置/后置/异常”三行就结束。

## 常见问题与注意事项

1. **这是让开发变慢吗？**  
   前期会慢一点，但需求迭代时明显更稳，返工更少。

2. **契约一定要靠类方法表达吗？**  
   不一定。函数式、API 层也可以表达契约；关键是语义清晰且可强制。

3. **是不是只要多写 if 就算契约？**  
   不是。契约必须包含“可依赖承诺”，尤其是明确的失败语义。

4. **接口文档写清楚就够了吗？**  
   不够。契约需要被代码和测试共同约束，不能只停留在注释。

5. **数据契约和参数校验（schema validation）是什么关系？**  
   参数校验只是数据契约的一部分。数据契约还包括单位、精度、默认值、兼容策略、字段语义等。

6. **副作用契约要写到多细？**  
   至少写清三件事：会改哪些外部状态、顺序要求、失败后的处理策略（回滚/重试/降级/告警）。

## 最佳实践与建议

- 每个新功能都先产出一页“目的 + 不变量 + 契约”草稿
- 核心行为拒绝静默失败（`return null/false` 需谨慎）
- 把“状态变化”收敛到少量核心模型方法
- 测试优先覆盖违约路径和边界条件
- 对核心用例至少显式写出这 6 类契约：数据 / 状态 / 不变式 / 行为 / 失败 / 副作用
- 在 PR 评审里加入一句固定问题：**“这个改动新增或改变了什么契约？”**

## 小结 / 结论

这句话的本质不是“先想清楚再写代码”这么泛，而是：

- 先定义系统必须成立的事实（不变量）
- 再定义可依赖的行为承诺（契约）
- 最后让实现去满足这些约束

当你这样做时，系统会从“实现驱动”转成“语义驱动”，可维护性会显著提升。

## 参考与延伸阅读

- Eric Evans, *Domain-Driven Design*
- Martin Fowler, *Patterns of Enterprise Application Architecture*
- Bertrand Meyer, *Object-Oriented Software Construction*（Design by Contract）

## 元信息

- **阅读时长**：7~9 分钟  
- **标签**：DDD、不变量、契约设计、工程实践  
- **SEO 关键词**：不变量, 契约, Evans, Fowler, DbC  
- **元描述**：用可运行示例解释“先定不变量与契约，再写实现”的工程意义与落地方法。

## 行动号召（CTA）

挑你当前项目里的一个核心方法，先不改实现，只先写出：

1. 它必须保护的不变量  
2. 它的数据契约（类型/范围/单位/可空性）  
3. 它的状态契约（允许哪些状态迁移）  
4. 它的行为与失败契约（成功保证 / 失败语义）  
5. 它的副作用契约（会改哪里、失败怎么处理）

然后再重写实现，你会立刻看到复杂度下降。
