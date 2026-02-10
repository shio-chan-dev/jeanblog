---
title: "先定不变量与契约，再写实现：Evans/Fowler 实战法"
date: 2026-02-11T07:53:37+08:00
draft: false
description: "解释“先定不变量/契约，再写实现”到底在工程上多了什么，并给出可执行的落地模板。"
tags: ["DDD", "不变量", "契约设计", "软件设计", "工程实践"]
categories: ["工程实践"]
keywords: ["不变量", "契约", "Evans", "Fowler", "Design by Contract"]
---

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

## 实践指南 / 步骤

1. **先写目的，不写实现**  
   明确本次功能要改变什么业务结果。
2. **列不变量清单**  
   逐条写出“绝对不能被破坏”的规则。
3. **定义契约**  
   为核心行为定义前置条件、后置条件、失败语义。
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

## 解释与原理

“先定不变量/契约，再写实现”并不等于“偏爱 OOP”。  
它真正解决的是责任分配：

- 没有契约时：调用方承担判断责任（读实现、猜结果、补防御）
- 有契约时：被调用方承担规则责任（成功保证、失败明确）

所以差别不在“有没有 class”，而在“调用方是否能闭眼依赖该行为”。

## 常见问题与注意事项

1. **这是让开发变慢吗？**  
   前期会慢一点，但需求迭代时明显更稳，返工更少。

2. **契约一定要靠类方法表达吗？**  
   不一定。函数式、API 层也可以表达契约；关键是语义清晰且可强制。

3. **是不是只要多写 if 就算契约？**  
   不是。契约必须包含“可依赖承诺”，尤其是明确的失败语义。

4. **接口文档写清楚就够了吗？**  
   不够。契约需要被代码和测试共同约束，不能只停留在注释。

## 最佳实践与建议

- 每个新功能都先产出一页“目的 + 不变量 + 契约”草稿
- 核心行为拒绝静默失败（`return null/false` 需谨慎）
- 把“状态变化”收敛到少量核心模型方法
- 测试优先覆盖违约路径和边界条件

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
2. 它的前置条件  
3. 它的后置条件  
4. 它的失败语义

然后再重写实现，你会立刻看到复杂度下降。
