---
title: "先写骨架，再补细节：用契约拆解算法题与中型程序"
subtitle: "把 helper、主流程、不变量和 DDD 边界放回各自该在的位置"
date: 2026-03-20T14:44:10+08:00
summary: "围绕“公开接口先行、helper 以契约占位、实现围绕不变量展开”这条主线，系统讲解如何从算法题过渡到中型程序设计，并用 LRUCache 与下单流程示例说明它和 DDD 的分工关系。"
description: "把复杂程序拆成骨架、契约和原子状态变换：本文用 LRUCache、下单流程与反例解释为什么先写外部结构通常更稳，以及 helper 应该如何控制副作用。"
tags: ["algorithms", "算法", "程序设计", "软件设计", "DDD", "LRU", "抽象", "工程实践"]
categories: ["逻辑与算法"]
keywords: ["先写骨架", "契约驱动", "helper 设计", "LRUCache", "自顶向下设计", "DDD"]
readingTime: 15
draft: false
---

## 标题

先写骨架，再补细节：用契约拆解算法题与中型程序

## 副标题 / 摘要

很多人写代码不稳，不是因为不会语法，而是因为一开始就把“公开行为”“内部细节”“副作用边界”揉成了一团。  
本文要讲的不是某一种框架，而是一套通用方法：先写外部骨架，先定 helper 的契约，再围绕不变量填实现。它既能用于 `LRUCache` 这类数据结构题，也能用于 `handle_order` 这类中型业务流程。

## 目标读者

- 刷题时经常把 `get` / `put` 写成一团，改一处坏三处的开发者
- 已经会写函数，但开始接触中型模块设计的初中级工程师
- 想弄清“自顶向下设计”和 DDD 到底是什么关系的人

## 背景 / 动机

大多数“代码越写越乱”的问题，根源不是缺少模式，而是**没有先分层思考**。典型症状有三类：

1. **公开方法里塞满细节**  
   比如一个 `put()` 里同时做哈希表查询、双向链表摘节点、尾部淘汰、头部插入、边界判断。逻辑能跑，但一次修改会牵动 2 个数据结构和 4 到 8 次指针赋值。
2. **helper 名字存在，契约不存在**  
   比如 `_sync()`、`_process()`、`_handle()` 这种名字，调用方不知道它改了哪些状态、失败时会怎样、是否有隐藏副作用。
3. **一上来就写局部细节，最后回头补结构**  
   结果通常是：越往后越不敢抽象，因为每个局部都已经偷偷依赖了别的局部。

用一个最常见的数字感受一下问题规模。假设你用“数组 + 线性扫描”的方式写 LRU：

- 容量 `n = 10^5`
- 调用次数 `q = 2 * 10^5`

最坏情况下，单次查找或调整顺序要 `O(n)`，总成本接近 `O(nq) = 2 * 10^10` 次比较或搬移。  
这时你会发现，真正要先定下来的不是某一行代码，而是**结构骨架**：必须是“哈希表 + 双向链表”，并且 `get` / `put` 都要把复杂度压到 `O(1)`。

这正是本文方法论的起点：  
**先决定外部行为和内部状态，再决定 helper；先决定 helper 的契约，再决定它的实现。**

## 快速掌握地图（60-120 秒）

- 问题形状：公开 API 很少，但内部状态变换很多，且多个方法会重复操作同一批状态
- 核心思想一句话：先写主流程骨架，再把可复用的原子状态变换抽成 helper，并为每个 helper 明确前置条件、后置条件和副作用
- 什么时候用：状态型数据结构、应用服务编排、需要维护不变量的中型模块
- 什么时候避免：需求本身还没搞清楚、领域概念变化超过 50% 的探索期原型
- 复杂度头条：像 `LRUCache` 这种结构题，目标通常是 `get/put = O(1)`；像业务流程题，目标通常是“主流程 5 到 15 行内能读懂”
- 常见失败模式：helper 表面上只“弹出旧节点”，实际上还偷偷删了 `map`，导致调用方二次删除时报错

## 深化焦点（PDKH）

本文只深化两个概念，不平行扩散：

1. **概念 A：契约先行的 helper 设计**
2. **概念 B：围绕不变量做状态变换**

对这两个概念，本文都会走完 PDKH 的主路径：

- P：重述问题
- D：给最小可运行例子
- K：给出不变量或前后置条件
- H：给出形式化描述、复杂度阈值、反例和工程现实

## 主心智模型

把一个复杂方法想成三层：

1. **公开层（orchestration layer）**  
   决定“这件事要分几步做”，例如 `put()`、`get()`、`handle_order()`
2. **原子变换层（primitive transitions）**  
   决定“每一步到底只改什么状态”，例如 `_remove(node)`、`_add_front(node)`
3. **不变量层（invariants）**  
   决定“无论怎么走步骤，哪些事实必须始终成立”

如果用函数组合的写法来表达，公开方法本质上是若干原子变换的组合：

`public_method = T_k ∘ T_(k-1) ∘ ... ∘ T_1`

其中：

- `T_i` 表示第 `i` 个原子状态变换
- 每个 `T_i` 都应该有清晰契约：`Contract(T_i) = (Pre_i, Effect_i, Post_i)`
- 公开方法的正确性，来自这些变换对不变量的逐步保持

这套模型既适用于算法题，也适用于业务代码：

- 在 `LRUCache` 里，`put()` 是编排层，`_remove()` / `_add_front()` 是原子变换层
- 在下单系统里，`place_order()` 是编排层，`reserve_inventory()` / `order.confirm()` 是原子变换层

这也是为什么“先写骨架”不是空谈。骨架不是 TODO 列表，而是**对变换顺序和状态边界的明确建模**。

## 核心概念与术语

- **骨架（skeleton）**：公开方法的步骤结构，不追求细节，但要能看出控制流和职责边界
- **契约（contract）**：一个方法对外承诺的输入要求、状态变化、输出结果和失败语义
- **helper**：被公开方法复用的局部操作，通常应该比公开方法更原子、更窄职责
- **不变量（invariant）**：在每次状态变化前后都必须成立的事实
- **副作用（side effect）**：对外部可观察状态的修改，例如改链表、删字典、写数据库、发消息
- **编排（orchestration）**：按顺序调用多个步骤完成一个完整用例，但不把每一步的细节都摊在主流程里

两个最重要的形式化表达：

1. helper 契约

   `Contract(h) = (Pre_h, Effect_h, Post_h)`

   其中：

   - `Pre_h`：调用前必须满足的条件
   - `Effect_h`：helper 会修改哪些状态
   - `Post_h`：调用后保证成立的条件

2. 结构一致性

   `|map| = number_of_real_nodes(list)`

   这个公式在 LRU 里尤其重要：

   - `|map|` 是哈希表里 key 的数量
   - `number_of_real_nodes(list)` 是双向链表里除哨兵外真实节点的数量

   如果这两个数字不相等，说明某个 helper 不是漏删了节点，就是漏删了映射。

## 可行性与下界直觉

### 为什么不能只靠“先写几个函数名”

如果你只写出：

- `get`
- `put`
- `_remove`
- `_add_front`
- `_pop_lru`

但没有决定内部状态结构，那骨架仍然是空的。  
例如只用数组来维护最近使用顺序，虽然也能写出这些函数名，但 `put` 和 `get` 无法稳定做到 `O(1)`。

这说明：  
**骨架不是只写 API 名字，而是先确定“公开行为 + 数据结构 + 不变量”。**

### 反例：骨架定太早，契约定太假

再看一个业务例子。你可能先写出：

```python
def handle_order(user_id: int, item_id: int) -> Receipt:
    user = load_user(user_id)
    item = load_item(item_id)
    validate_order(user, item)
    order = create_order(user, item)
    return build_receipt(order)
```

如果后来发现“下单”必须同时满足：

- 锁库存
- 校验优惠券
- 扣余额
- 写订单
- 发异步事件

那这个骨架仍然可以保留，但 helper 的契约必须更新。  
真正危险的不是“先写骨架”，而是**把骨架误当成真相，不再校正契约**。

## 问题建模

先把本文讨论的问题限定清楚：

### 场景 A：算法题 / 数据结构题

以 `LRUCache` 为例：

- 公开接口只有 2 个：`get(key)` 和 `put(key, value)`
- 核心状态有 2 组：哈希表 `map` 和双向链表 `list`
- 目标复杂度是：查询、更新、淘汰都要 `O(1)`

### 场景 B：中型程序 / 业务流程

以 `handle_order()` 为例：

- 公开接口可能只有 1 个用例：下单
- 依赖至少 3 类外部对象：用户仓储、商品仓储、订单仓储
- 目标不是极限性能，而是保持流程可读、职责边界清楚、失败路径可解释

### 共同优化目标

无论场景 A 还是 B，都希望做到三件事：

1. **主流程一眼能读懂**
2. **每个 helper 的副作用范围可预测**
3. **核心不变量能被逐条验证**

## 基线与瓶颈

### 朴素写法

很多人第一次写 LRU，会把所有逻辑直接塞进 `get` 和 `put`：

- 查字典
- 摘节点
- 接前驱后继
- 插到头部
- 满容量时淘汰尾部
- 维护映射

从渐进复杂度看，最后也许仍然能写到 `O(1)`；  
但从**正确性维护成本**看，它非常脆弱。

原因在于：一次双向链表的删除或插入，至少涉及 4 次指针修改。

例如 `_remove(node)` 的核心动作是：

1. `p.next = n`
2. `n.prev = p`

而 `_add_front(node)` 至少还有 4 次引用更新：

1. `node.prev = head`
2. `node.next = head.next`
3. `head.next.prev = node`
4. `head.next = node`

如果你把这些赋值分散在 3 个分支里复制粘贴，忘掉其中任意 1 次，链表就会断。  
这就是瓶颈：**不是不会写，而是重复的低层状态操作会污染高层控制流。**

### 业务代码的对应瓶颈

同样的问题会出现在业务代码里。假设 `place_order()` 里同时出现：

- 查用户
- 查商品
- 余额校验
- 库存校验
- 订单组装
- 仓储持久化
- 事件发送

如果所有逻辑都堆在一个 80 行函数里，任何规则变动都会迫使你重新在一大段流程里寻找边界。  
这时即便没有指针错误，也会出现**语义耦合错误**：一个“查询 helper”突然兼做了“写库”和“发事件”。

## 关键观察

复杂代码通常不是因为“事情太多”，而是因为**相同的状态变换没有被命名**。

对 `LRUCache` 来说，真正难的不是 `put()` 这 10 来行，而是你是否意识到下面几件事反复出现：

- 从链表中摘掉一个节点
- 把节点插到头部
- 找出尾部旧节点
- 把已有节点标记成最近使用

一旦这些动作被独立命名，主流程会突然变得清晰：

```python
def put(self, key: int, value: int) -> None:
    node = self.map.get(key)
    if node is not None:
        node.val = value
        self._move_front(node)
        return
    if len(self.map) == self.cap:
        old = self._pop_lru()
        del self.map[old.key]
    node = Node(key, value)
    self.map[key] = node
    self._add_front(node)
```

这段代码之所以可读，不是因为它短，而是因为它只表达**决策顺序**，不表达底层指针细节。

也就是说：

- 主流程应该负责“判断分支和拼装步骤”
- helper 应该负责“单一状态变换”
- 不变量应该负责“保证这些变换能安全组合”

## 算法步骤（实践指南）

下面给出一套通用流程，既能写算法题，也能写中型模块。

1. **先写公开行为**

   先回答：这个模块从外部看，真正要暴露什么？  
   例如 LRU 就是 `get/put`；下单服务就是 `place_order`。

2. **列出内部状态**

   不要急着写 helper，先写清楚你要维护哪几组状态。  
   LRU 是 `map + list`；下单流程可能是 `user/product/order + transaction/event`。

3. **定义不变量**

   每组状态之间应该满足什么关系？  
   例如：

   - `map` 和链表节点数量一致
   - 链表顺序必须是 MRU -> LRU
   - 一个订单在 `confirm()` 后状态必须从 `CREATED` 进入 `CONFIRMED`

4. **把重复动作抽成 helper，并先写契约**

   不需要一开始就实现，但至少要明确：

   - 调用前条件是什么
   - 会改哪些状态
   - 调用后保证什么

5. **用 helper 契约拼出主流程**

   此时主流程应该像“步骤清单”，而不是像“寄存器操作手册”。

6. **优先实现最原子的 helper**

   例如 LRU 里先实现 `_remove` 和 `_add_front`，再实现 `_move_front` 和 `_pop_lru`。

7. **用具体 trace 验证不变量**

   至少写一个最小非平凡用例。  
   LRU 可以用容量 `2` 的追踪；业务流程可以用“正常下单 + 库存不足”两条路径。

8. **最后才做局部优化**

   如果某个 helper 太短，不抽也可以；但前提是你已经确认副作用边界没有被破坏。

## 决策标准（选型指南）

这套方法不是万能钥匙，但适用面很广。下面给一组经验型判断。

### 适合优先写骨架的场景

- 公开 API 数量很少，通常在 `1` 到 `5` 个之间
- 每个公开方法要反复操作同一批状态对象，通常是 `2` 到 `5` 组
- 模块里存在明确不变量，例如顺序、计数、一致性、状态流转
- 你希望后续能单测 helper，而不是每次都靠整体验证

### 更适合先做探索的场景

- 需求本身还不稳定，今天是 `Coupon`，明天变成 `Campaign`
- 领域名词尚未收敛，名词表三天换一次
- 你连核心状态有哪些都说不清，只能先做快速原型

### 什么时候需要从“轻量骨架”升级到 DDD

- 同一个业务概念被 3 个以上用例重复引用
- 规则散在服务、控制器、仓储、任务消费者等多个位置
- 你发现“对象是什么”比“先调用哪个函数”更关键

一句话概括：

- **公开行为和状态变换已经清楚**：先写骨架
- **业务概念和边界还混乱**：先收敛领域模型，再谈骨架

## Worked Example（追踪）

用容量为 `2` 的 LRU 做最小非平凡追踪：

初始状态：

- `map = {}`
- 链表：`head <-> tail`

执行序列：

1. `put(1, 10)`
2. `put(2, 20)`
3. `get(1)`
4. `put(3, 30)`

逐步状态如下：

| 步骤 | 返回值 | 链表（从 MRU 到 LRU） | map 中的 key |
| --- | --- | --- | --- |
| 初始 | - | `[]` | `{}` |
| `put(1, 10)` | - | `[1]` | `{1}` |
| `put(2, 20)` | - | `[2, 1]` | `{1, 2}` |
| `get(1)` | `10` | `[1, 2]` | `{1, 2}` |
| `put(3, 30)` | - | `[3, 1]` | `{1, 3}` |

第 4 步最值得看，因为这里同时触发了两件事：

1. `len(map) == cap`，需要淘汰 LRU 节点 `2`
2. 新节点 `3` 插入头部成为 MRU

如果 `_pop_lru()` 只负责“弹链表尾部并返回节点”，`put()` 就可以自己显式做：

```python
old = self._pop_lru()
del self.map[old.key]
```

这就让副作用边界变得透明：  
**链表由 `_pop_lru()` 负责，哈希表由 `put()` 负责。**

## 正确性（证明草图）

继续以 LRU 为例，设三个核心不变量：

- `I1`：`head` 和 `tail` 始终是哨兵节点，不参与真实数据存储
- `I2`：`map` 中每个 `key` 恰好对应链表中的一个真实节点
- `I3`：链表从 `head.next` 到 `tail.prev` 的顺序始终表示“最近使用到最久未使用”

### 为什么 `_remove(node)` 保持不变量

前提是 `node` 已经在链表中。  
它只做两件事：

- 把前驱直接接到后继
- 把后继的 `prev` 回指到前驱

因此：

- 不会创建新节点，也不会复制旧节点，所以 `I2` 不被破坏
- 不会改变其他节点的相对顺序，只是删除当前节点，所以 `I3` 在剩余节点上仍成立

### 为什么 `_add_front(node)` 保持不变量

它把节点插到 `head` 后面：

- 不破坏哨兵结构，`I1` 仍然成立
- 节点只出现一次，因此不重复，`I2` 仍然成立
- 新插入节点成为最靠近 `head` 的真实节点，因此自然代表最新使用，`I3` 成立

### 为什么 `get()` 正确

- key 不存在时返回 `-1`，状态不变
- key 存在时，把节点移动到头部并返回值

由于 `_move_front(node) = _remove(node) + _add_front(node)`，它在保持 `I1/I2` 的同时，把该节点更新为最新使用，因此 `get()` 的语义正确。

### 为什么 `put()` 正确

分三种情况：

1. **已存在 key**  
   只更新值并挪到头部，不会产生重复节点
2. **不存在 key 且未满**  
   新建节点，插头部，新增映射
3. **不存在 key 且已满**  
   弹出尾部旧节点，删除映射，再插入新节点

每种情况都保持 `map` 与链表一一对应，因此 `put()` 正确。

## 复杂度

以本文的 LRU 实现为例：

- 时间复杂度
  - `get`：`O(1)`
  - `put`：`O(1)`
  - `_remove` / `_add_front` / `_move_front` / `_pop_lru`：`O(1)`
- 空间复杂度
  - `O(cap)`，其中 `cap` 是缓存容量

但本文真正关心的不只是渐进复杂度，还包括**认知复杂度**：

- 主流程只保留分支和步骤
- helper 只保留原子变换
- 不变量集中承担正确性解释

这会让你在规模从 30 行长到 300 行时，仍然知道“该去哪里改”。

## 常数项与工程现实

### 1. 哨兵节点不是花活，是边界消除器

如果没有 `head/tail` 哨兵，删除头节点、尾节点、唯一节点时都要额外分支。  
引入两个哨兵后，链表操作大幅统一，常数项里多 2 个节点，但少掉一整类条件判断。

### 2. helper 过短也值得抽，前提是它承载重复语义

`_remove()` 只有 2 行核心赋值，看起来很短；  
但它承载的是“从链表中摘节点”这个反复出现的语义。抽出来的价值，不在节省键盘，而在防止高层逻辑重复操作指针。

### 3. 不要为了省 1 次函数调用，把职责重新揉回去

有人会说：`_move_front()` 不就是 `_remove() + _add_front()`，直接写进 `get()` 不更快？  
在 Python 里，少 1 次函数调用的收益，通常远小于职责边界被破坏后的排障成本。

### 4. 可以用现成库，但要知道它替你维护了什么

例如 Python 的 `OrderedDict` 可以很快写出一个 LRU。  
这在工程里完全合理，但前提是你知道：

- 它替你维护了顺序
- 你仍然要维护 key 的语义和淘汰时机
- 如果要面试、教学或自定义副作用，它不再帮你解释不变量

## 可运行实现（Language: Python）

下面给出一个完整可运行版本。注意看注释：它们不是解释语法，而是明确 helper 契约。

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    key: int = 0
    val: int = 0
    prev: "Node | None" = None
    next: "Node | None" = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map: dict[int, Node] = {}
        self.head = Node()  # MRU side sentinel
        self.tail = Node()  # LRU side sentinel
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Pre: node is already in the linked list.
        Effect: detach node from list only.
        Post: remaining list stays connected.
        """
        prev_node, next_node = node.prev, node.next
        assert prev_node is not None and next_node is not None
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_front(self, node: Node) -> None:
        """Pre: node is detached.
        Effect: insert node right after head only.
        Post: node becomes the MRU node.
        """
        first = self.head.next
        assert first is not None
        node.prev = self.head
        node.next = first
        self.head.next = node
        first.prev = node

    def _move_front(self, node: Node) -> None:
        self._remove(node)
        self._add_front(node)

    def _pop_lru(self) -> Node:
        """Effect: remove and return the LRU node from list only."""
        node = self.tail.prev
        assert node is not None and node is not self.head
        self._remove(node)
        return node

    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._move_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return

        node = self.map.get(key)
        if node is not None:
            node.val = value
            self._move_front(node)
            return

        if len(self.map) == self.cap:
            old = self._pop_lru()
            del self.map[old.key]

        node = Node(key, value)
        self.map[key] = node
        self._add_front(node)

    def snapshot(self) -> list[tuple[int, int]]:
        cur = self.head.next
        out = []
        while cur is not None and cur is not self.tail:
            out.append((cur.key, cur.val))
            cur = cur.next
        return out


if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 10)
    cache.put(2, 20)
    print(cache.snapshot())  # [(2, 20), (1, 10)]
    print(cache.get(1))      # 10
    print(cache.snapshot())  # [(1, 10), (2, 20)]
    cache.put(3, 30)
    print(cache.snapshot())  # [(3, 30), (1, 10)]
    print(cache.get(2))      # -1
```

## 工程场景

### 场景 A：刷题或面试里的数据结构题

这种场景最适合练“骨架 + 契约 + 不变量”。

```python
class LRUCache:
    def get(self, key: int) -> int:
        ...

    def put(self, key: int, value: int) -> None:
        ...
```

先把公开 API 写出来，再决定：

- 需要哪些状态
- 需要哪些 helper
- 每个 helper 改什么

### 场景 B：应用层 use case 编排

这里主流程更像 orchestration，helper 更像仓储或领域行为。

```python
def place_order(user_id: int, item_id: int) -> Receipt:
    user = user_repo.get(user_id)
    item = product_repo.get(item_id)
    order = Order.create_for(user)
    order.add_item(item, qty=1)
    order.confirm()
    order_repo.save(order)
    return Receipt.from_order(order)
```

这时要注意：

- 主流程只编排，不替每个对象承载业务细节
- 真正的规则应尽量沉到 `Order`、`InventoryPolicy` 这类对象里

### 场景 C：解析/校验/转换管线

很多脚本会把解析、校验、转换、输出写在一个函数里，其实也适合用同样方法拆开。

```python
def run_pipeline(raw: str) -> dict:
    tokens = tokenize(raw)
    ast = parse(tokens)
    validate(ast)
    return lower_to_ir(ast)
```

这里的 helper 契约要回答：

- `parse` 是否保证返回合法 AST
- `validate` 抛错还是返回错误集合
- `lower_to_ir` 是否会改输入对象

## Alternatives / Tradeoffs（替代方案与取舍）

### 方案 1：纯自底向上，想到哪写到哪

优点：

- 上手快
- 适合一次性脚本和探索期原型

缺点：

- 当公开方法要同时操作 `2` 个以上状态对象时，很快会失去边界
- 局部实现越多，后面越难反推出正确骨架

### 方案 2：重度前期建模，先把所有 helper、接口、类图一次性定死

优点：

- 文档很完整
- 大团队协作时便于提前对齐

缺点：

- 如果需求尚不稳定，前期模型会反复推倒
- 容易把“骨架”写成“僵化设计”

### 方案 3：轻量骨架 + 明确契约 + 迭代校正

这通常是最实用的平衡点：

- 先定公开行为和状态边界
- helper 可以先占位，但契约不能模糊
- 随着例子和反例增加，再校正实现和模型

### 这和 DDD 是什么关系

它们不是对立关系，而是关注点不同：

- **自顶向下骨架**回答的是：主流程如何拆步骤、谁调用谁
- **DDD**回答的是：业务里真正稳定的对象、边界和规则是什么

更准确地说：

- 当你已经知道公开用例要怎么走时，骨架法很有效
- 当你还不知道 `Order`、`Coupon`、`Inventory` 谁该拥有规则时，DDD 更重要

所以现实里通常是并存的：

1. 先用骨架法写出 use case 流程
2. 再用 DDD 把流程里的业务规则沉到合适对象
3. 最后让应用层只保留编排

## 迁移路径（Skill Ladder）

如果你已经能用这套方法稳定写出 `LRUCache` 这类题，下一步建议按这个顺序升级：

1. **不变量思维**  
   不只是“能跑”，而是能说明为什么每一步不破坏结构
2. **契约测试**  
   不只测公开 API，也测 helper 的前后置条件
3. **领域建模**  
   学会区分 orchestration、entity、repository、domain service
4. **并发与事务边界**  
   当状态不再只在内存里，helper 的副作用会扩展到锁、事务、消息、重试

更难的一类问题，是分布式系统里的流程编排：

- `reserve_inventory`
- `charge_payment`
- `create_order`
- `publish_event`

这时“helper 是否有隐藏副作用”会直接变成“系统是否会重复扣款”。

## 常见坑与边界情况

### 1. helper 名字明确，契约却模糊

例如 `_pop_lru()` 这个名字看起来已经挺清楚，但仍然可能有两种完全不同的实现：

- 只从链表中弹出旧节点
- 同时从链表和 `map` 里一起删掉

如果名字相同、契约不同，调用方就会踩坑。

### 2. helper 偷偷做了额外副作用

例如一个名叫 `load_user()` 的函数，除了查用户，还顺手：

- 刷新最后访问时间
- 写审计日志
- 预加载订单列表

这类函数短期省事，长期会让主流程无法预测成本和行为。

### 3. 过度拆分 micro-helper

如果一个 12 行函数被拆成 9 个 helper，每个 helper 只做一行，主流程反而失去局部连贯性。  
抽 helper 的标准不是“越短越好”，而是“是否代表稳定且可复用的语义”。

### 4. 只有骨架，没有验证

下面这种代码看起来结构很好，但没有任何信息量：

```python
def solve():
    prepare()
    process()
    finalize()
```

如果你说不清：

- `prepare` 产出什么状态
- `process` 依赖什么前置条件
- `finalize` 是否有持久化副作用

那这不是设计，而只是把未知包装成了名字。

## 最佳实践

- 先写公开 API，再写 helper；不要一上来就在局部细节里打转
- helper 的名字要体现语义，契约要体现副作用范围
- 主流程尽量只保留“判断 + 编排”，底层状态操作下沉
- 明确写出 2 到 3 条核心不变量，再围绕它们检查 helper
- 至少准备 1 个最小非平凡 trace 和 1 个失败反例
- 当业务对象比控制流更重要时，及时引入 DDD 视角重建模型

## 小结 / Takeaways

1. **先写骨架**不是先写空壳，而是先写清公开行为、状态结构和不变量。
2. **helper 可以先占位，但契约不能模糊**；否则只是把未知藏进了函数名。
3. **主流程应该表达决策顺序，helper 应该表达原子状态变换**；两者不要互相抢职责。
4. **DDD 和自顶向下骨架法不是二选一**；前者解决业务模型边界，后者解决实现流程拆分。
5. **凡是会反复改动多个状态对象的模块，都值得先做契约化拆解**；LRU 是最小练习场，业务流程是自然延伸。

## 参考与延伸阅读

- George Pólya, *How to Solve It*
- Edsger W. Dijkstra, *A Discipline of Programming*
- C. A. R. Hoare, *An Axiomatic Basis for Computer Programming*
- Eric Evans, *Domain-Driven Design*
- Martin Fowler, *Patterns of Enterprise Application Architecture*

## 行动号召（CTA）

找一段你最近写过的代码，最好满足下面两个条件：

- 公开方法不超过 `3` 个
- 内部至少同时维护 `2` 组状态

按本文方法重写一次：

1. 先写公开骨架
2. 再写 helper 契约
3. 最后补实现和 trace

如果你重写完发现主流程突然变短了，但解释能力变强了，说明你已经抓到这套方法的关键了。
