---
title: "以业务对象为核心的 Python 架构实践"
subtitle: "先有业务对象，再有接口、仓储和服务"
date: 2025-12-01T11:00:00+08:00
draft: false
tags: ["Python", "架构设计", "DDD", "FastAPI", "仓储模式"]
categories: ["Architecture", "Backend"]
summary: "这篇文章从一个简单的工单系统出发，展示如何在 Python 项目中以业务对象为中心设计接口、仓储与服务，而不是让 ORM、框架和表结构牵着鼻子走。"
description: "通过一个工单（Ticket）示例，讲解在 Python + FastAPI + SQLAlchemy 项目中，如何先定义清晰的业务对象（领域模型），再围绕它设计接口 DTO、仓储抽象和应用服务，避免业务代码直接围绕表结构和框架 API 打转。"
keywords: ["Python 业务对象", "领域模型设计", "仓储模式", "FastAPI 架构", "DDD 实践"]
readingTime: 12
---

> 本文想传达一个简单的观点：  
> **在 Python 项目中，一切都应该从“业务对象”开始，而不是从数据库表、ORM 模型或接口 JSON 开始。**

我们以一个极其常见的场景——**工单（Ticket）系统**——为例，演示如何：

- 先定义业务对象（领域模型）；
- 再围绕它设计接口层的 DTO；
- 再设计仓储抽象（Repository）；
- 最后再补上 Service 层和具体的数据库实现。

---

## 目标读者

- 使用 Python（尤其是 FastAPI / Flask）做业务开发的同学
- 对“代码结构越来越乱、改个字段要全项目找引用”感到疲惫的人
- 想从“表驱动 / JSON 驱动”逐步过渡到**以业务对象为核心**设计的后端工程师

---

## 背景：为什么“先表结构 / 先接口 JSON”容易失控？

在很多项目里，一个新需求的典型流程是：

1. 先画接口文档（Swagger/Apifox）；
2. 然后设计数据库表结构；
3. 再按表结构生成 ORM 模型；
4. Controller 里直接拿 ORM 当业务对象用；
5. 业务逻辑散落在 Controller / ORM / Service / SQL 里。

短期内很快，长期有几个典型问题：

- **业务概念被表结构绑死**：一旦表结构有历史包袱，新需求都要绕着旧表结构打补丁；
- **接口 DTO = ORM = 业务对象**：一个字段改名，要修改接口、表、代码一大圈；
- **测试困难**：没有清晰的“业务对象”，只能靠集成测试+真数据库。

而我们想要的是：

> 先想清楚“业务世界”里有什么对象，它们长什么样、有哪些行为，  
> 再考虑“这些对象要通过什么接口暴露出去”、“要存到什么表里”。

---

## 核心理念：一切从业务对象（领域模型）开始

所谓“以业务对象为核心”，可以粗暴地理解为：

1. 每个核心业务场景，都应该有对应的**领域模型**（Domain Model）；  
2. 领域模型**不依赖**框架、不依赖 ORM、不关心 HTTP 细节；  
3. 接口 DTO、仓储、Service、ORM，全是围绕这个模型展开的“适配层”。  

这跟经典的 DDD 完整体系还有差距，但足以让项目结构从“表驱动 CRUD”升级到“领域对象驱动”。

下面用一个“工单（Ticket）”场景开刀。

---

## 第一步：定义业务对象（领域模型）

假设需求是这样的：

- 工单包含标题、描述、状态（待处理/处理中/已完成）、优先级、创建时间、最后更新时间；
- 工单可以被指派给某个处理人；
- 后续可能扩展标签、评论、附件等。

我们先不管表、不管接口，先写“业务世界里的 Ticket”：

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Ticket:
    id: str
    title: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    creator_id: str
    assignee_id: Optional[str]
    created_at: int
    updated_at: int

    def start_progress(self, assignee_id: str) -> None:
        """开始处理工单：设置处理人并将状态置为处理中。"""
        self.assignee_id = assignee_id
        self.status = TicketStatus.IN_PROGRESS
        self.updated_at = int(time.time())

    def resolve(self) -> None:
        """将工单标记为已完成。"""
        self.status = TicketStatus.RESOLVED
        self.updated_at = int(time.time())
```

几点观察：

- 这个 `Ticket` 不关心数据库，不继承任何 ORM 基类；
- 行为（`start_progress` / `resolve`）**挂在业务对象自身上**，而不是散在 Controller 里；
- 将来如果换框架（FastAPI → Flask）或换数据库（SQLite → MySQL），这个类可以完全不动。

---

## 第二步：围绕业务对象设计接口 DTO

在有了 `Ticket` 之后，我们再反过来思考接口层：

- 接口需要哪些字段？
- 哪些字段是只读的（比如 `created_at`）？
- 哪些字段是客户端输入的？

可以用 Pydantic 定义 API 层的 Request / Response 模型：

```python
from pydantic import BaseModel
from typing import Optional


class CreateTicketRequest(BaseModel):
    title: str
    description: str
    priority: TicketPriority = TicketPriority.MEDIUM


class TicketResponse(BaseModel):
    id: str
    title: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    creator_id: str
    assignee_id: Optional[str]
    created_at: int
    updated_at: int

    @classmethod
    def from_domain(cls, ticket: Ticket) -> "TicketResponse":
        return cls(
            id=ticket.id,
            title=ticket.title,
            description=ticket.description,
            status=ticket.status,
            priority=ticket.priority,
            creator_id=ticket.creator_id,
            assignee_id=ticket.assignee_id,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
        )
```

接口层做的是：

- 把 HTTP 世界的 JSON 转成领域世界的 `CreateTicketRequest`；
- 调用 Service / 仓储拿到 `Ticket`；
- 用 `TicketResponse.from_domain` 包装成返回值。

---

## 第三步：为业务对象设计仓储抽象（Repository）

有了业务对象之后，仓储只需要回答一个问题：

> “我怎么把 Ticket 读出来 / 写回去？”

先定义仓储接口，不管具体怎么实现：

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class TicketRepository(ABC):
    """Ticket 的持久化抽象，返回/接收的都是 Ticket 领域对象。"""

    @abstractmethod
    def get(self, ticket_id: str) -> Optional[Ticket]:
        ...

    @abstractmethod
    def list(
        self,
        page: int,
        page_size: int,
        status: Optional[TicketStatus] = None,
    ) -> Tuple[List[Ticket], int]:
        ...

    @abstractmethod
    def save(self, ticket: Ticket) -> None:
        """创建或更新 Ticket。"""
        ...
```

上层完全不关心“用的 SQLite 还是 MySQL、SQLAlchemy 还是 raw SQL”，  
只要有一个对象满足 `TicketRepository` 的接口就行。

你可以：

- 写一个 `InMemoryTicketRepository` 做单测 / demo；
- 写一个 `SqlAlchemyTicketRepository` 做生产使用。

---

## 第四步：围绕业务对象设计 Service 层

Service 层的职责可以简单理解为：

> 组合多个业务对象和仓储，执行一个完整的业务用例。

比如“创建工单并自动分配默认处理人”：

```python
import time
import uuid
from typing import Optional


class TicketService:
    def __init__(self, repo: TicketRepository) -> None:
        self.repo = repo

    def create_ticket(
        self,
        creator_id: str,
        req: CreateTicketRequest,
        default_assignee_id: Optional[str] = None,
    ) -> Ticket:
        now = int(time.time())
        ticket = Ticket(
            id=uuid.uuid4().hex,
            title=req.title,
            description=req.description,
            status=TicketStatus.OPEN,
            priority=req.priority,
            creator_id=creator_id,
            assignee_id=None,
            created_at=now,
            updated_at=now,
        )

        if default_assignee_id:
            ticket.start_progress(default_assignee_id)

        self.repo.save(ticket)
        return ticket
```

这里有几个关键点：

- Service 接收的也是**业务对象或 DTO**，调用的是 `Ticket` 上的方法（行为）；
- Service 不关心 HTTP，不关心 ORM，只依赖 `TicketRepository` 抽象；
- Service 可以很容易被单元测试：传入一个 Fake 仓储就行。

---

## 第五步：接口层只是“适配器”，围绕业务对象展开

最后才轮到 Controller（以 FastAPI 为例）：

```python
from fastapi import APIRouter, Depends

router = APIRouter()


def get_ticket_service() -> TicketService:
    # 实际项目中可以通过依赖注入管理
    repo = SqlAlchemyTicketRepository(...)
    return TicketService(repo)


@router.post("/tickets", response_model=TicketResponse)
async def create_ticket_endpoint(
    req: CreateTicketRequest,
    current_user_id: str = Depends(...),
    service: TicketService = Depends(get_ticket_service),
):
    ticket = service.create_ticket(
        creator_id=current_user_id,
        req=req,
    )
    return TicketResponse.from_domain(ticket)
```

可以看到：

- 接口不再直接操作 ORM，不再直接写 SQL；
- 接口只是“HTTP 世界”和“领域世界”的适配层；
- 核心逻辑在 Ticket / TicketService / TicketRepository 这一条链路上。

---

## 与“每表一个 DAO + 大 Service”相比的取舍

很多项目的常见模式是：

- 每张表一个 DAO；
- Service 里注入一堆 DAO；
- Service 既负责业务流程，又写了大量 `session.query(...)`。

问题在于：

- Service 很容易变成“大泥球”：既懂表结构，又懂业务细节；
- 业务对象没有清晰边界：任何地方都在 new dict/list 拼数据；
- 很难做到“换存储实现而不影响业务代码”。

而本文这种“业务对象优先”的方式：

- 业务对象 (`Ticket`) 作为**中心抽象**，统一承载状态和行为；
- 仓储负责“怎么把 Ticket 存起来”，可以有多种实现；
- Service 负责“用 Ticket 完成一个业务用例”；
- 接口只是适配层，负责 JSON ↔ 业务对象的互转。

取舍在于：

- 你多写了一点“模型”和“接口”，但换来了更清晰的边界和更易维护的结构；
- 初期可能看起来“啰嗦”，但在需求越来越多时，收益会越来越明显。

---

## 常见问题与注意事项

**Q1：业务对象和 ORM 模型可以是同一个类吗？**  
可以，但不建议。  
ORM 通常关注的是“表结构 + 关系 + 性能”，而业务对象关注的是“行为 + 不变量”。长期来看，分离更健康。

**Q2：Service 一定要有吗？能不能 Controller 直接用仓储？**  
小项目可以，但随着需求复杂，很快 Controller 会堆满业务逻辑。  
Service 是承载“用例”的天然落点，值得保留。

**Q3：领域模型要不要一开始就设计得很复杂？**  
不用。一开始可以很简，随着需求演化再拆 Value Object / 子聚合。  
关键是“有一个相对稳定的地方来承载业务概念”，而不是满世界 dict。

**Q4：Fake 仓储是不是浪费时间？**  
相反，它非常实用：

- 本地可以不用连数据库就跑通大部分逻辑；
- 单元测试可以只依赖内存实现；
- 切换真实仓储时，业务代码可以不用动。

---

## 最佳实践小结

- **任何新模块，先写业务对象（领域模型），再考虑表和接口。**
- 使用 `dataclass` / 枚举等原生手段建模，不要一上来就绑死在 ORM 上。
- 接口层的模型（Pydantic）只负责输入/输出校验和序列化，领域模型负责行为。
- 仓储只关心“如何持久化领域对象”，不要泄漏 ORM/SQL 到业务层。
- Service 负责完整的业务用例，组合多个业务对象和仓储。
- 多用 Fake 仓储支撑开发和测试，真实实现可以后置。

---

## 小结与下一步

这篇文章用一个简单的工单系统例子，展示了“以业务对象为核心”的一条路径：

1. 先定义领域模型 `Ticket` 及其行为；
2. 围绕它设计接口 DTO（Pydantic 模型）；
3. 定义 `TicketRepository` 抽象，隐藏存储细节；
4. 用 `TicketService` 封装完整用例；
5. 最后再在接口层做适配。

如果你手上有一个正在维护的项目，可以尝试：

- 先挑一个子模块（比如“权限组管理”、“工单管理”），  
  按上面的步骤抽出一个业务对象 + 仓储 + Service；
- 保持对其他模块的侵入尽量小，逐步迁移，不必一次性“大重构”。

---

## 参考与延伸阅读

- Eric Evans，《领域驱动设计：软件核心复杂性应对之道》
- Vaughn Vernon，《实现领域驱动设计》
- Martin Fowler: Anemic Domain Model / Rich Domain Model
- FastAPI 官方文档：关于依赖注入与测试部分
- SQLAlchemy / Alembic 官方文档：模型与迁移

---

## 行动号召（CTA）

- 回到你当前的项目里，挑一个“最核心的业务概念”，尝试给它写一个独立的 `@dataclass` 领域模型。
- 围绕这个模型画一张小图：接口 DTO、仓储、Service 各自应该怎么依赖它。
- 如果你愿意，可以把你设计的业务对象和依赖关系贴出来，我们可以一起 review 一下，看看还能如何优化边界划分。

