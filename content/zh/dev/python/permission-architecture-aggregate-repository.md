---
title: "从表结构到领域模型：用聚合仓储设计权限系统"
subtitle: "为什么你的业务代码里看不到数据库表，是一件好事"
date: 2025-12-01T10:00:00+08:00
draft: false
tags: ["Python", "FastAPI", "DDD", "仓储模式", "权限系统"]
categories: ["Architecture", "Backend"]
summary: "以一个权限组管理模块为例，展示如何用领域模型 + 聚合仓储的方式设计后端，而不是让业务直接围着数据库表转。"
description: "本文通过 FastAPI + SQLAlchemy + Alembic 的权限组管理示例，讲解如何用领域模型和仓储抽象设计权限系统，为什么仓储可以一次操作多张表，以及 service 层和 repository 层各自应该承担的职责。"
keywords: ["Python 权限系统", "领域驱动设计", "聚合仓储", "仓储模式", "FastAPI 架构设计"]
readingTime: 12
---

> 以一个“权限组管理”模块为例，聊聊**表结构、领域模型、仓储、Service**之间该怎么划分边界，回答两个常见问题：
>
> 1. 为什么业务代码里看不到任何表结构的影子？
> 2. 一个仓储一次操作四张表，是不是“耦合过重、设计很脏”？

---

## 目标读者

- 使用 **Python + FastAPI + SQLAlchemy + Alembic** 做业务开发的同学
- 希望慢慢从 “表驱动 CRUD” 进化到 **更清晰的分层和领域模型** 的后端工程师
- 对 **DDD（领域驱动设计）中的仓储模式 / 聚合根** 有兴趣，但不想被大量理论劝退的人

---

## 背景与动机：为什么“看不到表结构”反而是好事？

在很多项目里，业务代码长这样：

- Controller 里直接 `session.query(Table).filter(...).all()`
- Service 里全是 `db.execute(...)`、`join`、`分页 + 条件拼接`
- 改个字段要从 Controller 一路改到 SQL

用久了会发现几个痛点：

- 业务逻辑和存储细节强耦合，**改表结构 = 全项目地震**
- 很难写 Fake 实现做测试，本地 demo 也必须连数据库
- 权限这一类跨多表的功能（组、用户、权限点），逻辑散落在各个地方

于是就有了一个很常见的问题：

> “我现在的业务模型里，完全看不到表结构的痕迹，是不是设计错了？”

答案通常是：**没错，反而说明你在向“领域层”和“仓储抽象”靠近**。

接下来我们用一个权限组管理的真实例子，把这件事讲清楚。

---

## 核心概念：领域模型 vs 仓储 vs DAO vs Service

先把几个关键词说白：

- **领域模型（Domain Model）**  
  描述业务世界的概念，比如 `PermissionGroup`、`GroupMember`、`Permission`，只关心业务属性和规则，不关心怎么存到数据库。

- **仓储（Repository / Table Abstraction）**  
  把“如何把一个领域对象存取到某种存储（DB、内存、Redis）”封装起来，对外只暴露领域模型。  
  在你的代码里就是 `BasePermissionTable` / `AbstractUserTable` 这一层。

- **DAO / 每表一个小仓储**  
  常见于 CRUD 项目：`UserDAO`、`RoleDAO`、`PermissionDAO`……每个类只管一张表的 CRUD，对业务一无所知。

- **聚合根（Aggregate Root）**  
  一个**业务上天然绑在一起**的对象集合，比如“权限组 + 成员列表 + 权限树”，对外以一个整体保存/加载。

- **Service（应用服务 / 领域服务）**  
  更偏业务编排：执行业务流程、调用多个仓储、做权限校验、发送事件等，而不是操作 SQL 细节。

**关键区别：**

- DAO 是“围着表转”的；
- 仓储是“围着领域模型/聚合转”的；
- Service 则是站在业务视角 orchestrate。

---

## 示例场景：权限组管理的领域模型

先看一组精简版的领域模型（与表结构完全解耦）：

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PermissionGroup:
    id: str
    name: str
    user_count: int
    created_at: int
    updated_at: int
    description: Optional[str] = None
    built_in: bool = False


@dataclass
class GroupMember:
    user_id: str
    name: str
    role: Optional[str] = None
    in_group: bool = False


@dataclass
class Permission:
    module: str
    code: str
    label: str
    checked: bool = False


@dataclass
class PermissionGroupDetail:
    group: PermissionGroup
    members: List[GroupMember]
    permissions: List[Permission]


@dataclass
class SavePermissionGroupCommand:
    group_id: Optional[str]
    name: Optional[str]
    description: Optional[str]
    user_ids: List[str]
    permission_codes: List[str]
```

注意几点：

- 这里**完全不知道数据库长什么样**，也没出现任何 ORM/Session。
- `PermissionGroupDetail` 是一个典型的**聚合根**：一个权限组 + 其成员 + 权限树。

---

## 仓储抽象：BasePermissionTable 只说“我要什么”，不说“怎么查”

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from domain.permission_group import (
    PermissionGroup,
    PermissionGroupDetail,
    SavePermissionGroupCommand,
)


class BasePermissionTable(ABC):
    """
    权限组仓储抽象：返回领域模型，而不是 ORM。
    """

    @abstractmethod
    def list_groups(
        self,
        page: int,
        page_size: int,
    ) -> Tuple[List[PermissionGroup], int]:
        ...

    @abstractmethod
    def get_detail(
        self,
        group_id: Optional[str],
    ) -> Optional[PermissionGroupDetail]:
        ...

    @abstractmethod
    def save(
        self,
        command: SavePermissionGroupCommand,
    ) -> str:
        """返回保存后的 group_id"""
        ...

    @abstractmethod
    def delete(self, group_id: str) -> bool:
        ...
```

**要点：**

- 上层（Controller / Service）只依赖这个接口和领域模型；
- 底层可以有很多种实现：内存 Fake、MySQL、SQLite、甚至远程服务。

---

## FakePermissionTable：不用数据库的“内存实现”

用内存字典做一个假的实现，在本地开发 / 单测里非常好用：

```python
class FakePermissionTable(BasePermissionTable):
    def __init__(self) -> None:
        self._groups: Dict[str, PermissionGroupDetail] = {}
        self._init_memory()

    def list_groups(self, page: int, page_size: int) -> Tuple[List[PermissionGroup], int]:
        all_groups = [detail.group for detail in self._groups.values()]
        total = len(all_groups)
        start = (page - 1) * page_size
        end = start + page_size
        return all_groups[start:end], total

    def get_detail(self, group_id: Optional[str]) -> PermissionGroupDetail:
        # 如果存在，直接返回
        if group_id and group_id in self._groups:
            return self._groups[group_id]

        # 不存在时，返回一个“新建模板”
        ...
```

这里你已经可以看到**好处**：

- Controller 调 `permission_table.get_detail(...)` 时，不知道背后是内存还是数据库；
- 用 `FakePermissionTable` 做 e2e 测试时，连数据库都不需要。

---

## 真实表结构：4 张表支撑一个聚合

当你要上真实数据库时，就需要设计表结构。一个合理的拆分是 4 张表：

1. `permission_group`：权限组定义
2. `permission_def`：权限点定义（code / module / label）
3. `permission_group_user`：权限组 ↔ 用户关系
4. `permission_group_permission`：权限组 ↔ 权限点关系

它们是**储存细节**，属于“基础设施层”，不应该蔓延到 Controller / Domain 层。

---

## 为什么一个仓储可以操作四张表，而不是“太耦合”？

回到常见疑问：

> “一个数据库交互层同时对四个表进行了操作，我是不是应该把四个表的操作分开，然后把这个整体的逻辑放在 services 中？”

拆开看：

- **领域上**：权限组详情（PermissionGroupDetail）本来就跨 3 类信息：组、成员、权限树。
- **保存** 一个权限组时，业务上希望：
  - 组基础信息更新；
  - 成员列表整体替换；
  - 权限勾选整体替换；
  - 这些要么都成功，要么都回滚——典型的**一个事务 / 一个聚合**。

从这个角度，写一个 `SqlPermissionTable`，在一个方法里操作 3～4 张表，是**很自然的聚合仓储**，而不是坏耦合。

如果你把这些表的操作全部拆到不同 DAO 里，再让 Service 去 orchestrate：

- Service 里既有业务规则，又有各种 join 和 transaction 细节；
- 如果不小心在多个 DAO 里各自开 session/事务，数据一致性还更难保证；
- 本质上是把“复杂度”从仓储挪到 Service，**并没有减少耦合，只是换了地方**。

> 更合理的边界是：  
> **仓储对一个“聚合”负责（可以内部动多张表）**，  
> Service 对“业务流程 / 多个聚合之间的编排”负责。

---

## 示例：SqlPermissionTable 的大致结构（精简版）

下面是一个精简版本的 `SqlPermissionTable`，用来展示如何在一个仓储里操作多张表，但对外只暴露领域模型：

```python
class SqlPermissionTable(BasePermissionTable):
    """
    基于数据库的权限组仓储实现。

    - 对外：PermissionGroup / PermissionGroupDetail / Command
    - 对内：PermissionGroupORM + PermissionGroupUserORM + PermissionGroupPermissionORM + PermissionDefORM
    """

    def list_groups(self, page: int, page_size: int) -> Tuple[List[PermissionGroup], int]:
        with get_db() as session:
            query = session.query(PermissionGroupORM)
            total = query.count()
            rows = (
                query
                .order_by(PermissionGroupORM.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

            groups = [self._to_domain_group(row) for row in rows]
            return groups, total

    def get_detail(self, group_id: Optional[str]) -> Optional[PermissionGroupDetail]:
        if not group_id:
            return None

        with get_db() as session:
            group_row = (
                session.query(PermissionGroupORM)
                .filter(PermissionGroupORM.id == group_id)
                .one_or_none()
            )
            if not group_row:
                return None

            group = self._to_domain_group(group_row)

            # 成员
            member_rows = (
                session.query(PermissionGroupUserORM)
                .filter(PermissionGroupUserORM.group_id == group_id)
                .all()
            )
            members = [
                GroupMember(user_id=m.user_id, name=m.user_id, role=m.role, in_group=True)
                for m in member_rows
            ]

            # 权限：所有权限定义 + 是否勾选
            perm_defs = session.query(PermissionDefORM).all()
            group_perm_rows = (
                session.query(PermissionGroupPermissionORM.permission_code)
                .filter(PermissionGroupPermissionORM.group_id == group_id)
                .all()
            )
            group_codes = {row.permission_code for row in group_perm_rows}

            permissions = [
                Permission(
                    module=p.module,
                    code=p.code,
                    label=p.label,
                    checked=p.code in group_codes,
                )
                for p in perm_defs
            ]

            return PermissionGroupDetail(
                group=group,
                members=members,
                permissions=permissions,
            )

    def save(self, command: SavePermissionGroupCommand) -> str:
        now = int(time())
        with get_db() as session:
            group_id = command.group_id or self._gen_group_id()

            # 1. upsert group
            ...

            # 2. 重建组成员关系
            ...

            # 3. 重建组权限关系
            ...

            session.commit()
            return group_id

    def delete(self, group_id: str) -> bool:
        with get_db() as session:
            ...
```

这里的“耦合”是：

- 对领域：一个仓储负责一个聚合，是**合理、期望中的耦合**；
- 对数据库：仓储内部确实知道 3～4 张表，但这些细节没有泄漏到 Controller/Service/Domain。

---

## Service 应该负责什么、而不是负责什么？

结合一个典型的 FastAPI 项目，可以大致分层：

- **Controller（FastAPI 路由）**  
  - 解析 HTTP 请求（JSON、Query、Header）
  - 调用 Service / 仓储
  - 组装成统一响应模型（`UnifiedResponse`）

- **Service（应用服务 / 领域服务）**  
  适合做：
  - 跨多个聚合的业务流程（比如：创建用户 + 加入默认权限组 + 发送欢迎消息）
  - 权限校验、业务规则判断（比如：某些组只能管理员修改）

  不适合做：
  - 不断写 `session.query(...)` 跟表打交道；
  - 管理具体事务边界和 SQL 细节（这应该在仓储里）。

- **Repository（仓储 / Table 抽象）**  
  - 对一个聚合根负责读写；
  - 可以动多张表，但对上层隐藏存储细节；
  - 可以有 Fake 实现和真实实现。

- **ORM / 数据库 / Alembic**  
  - 定义表结构和迁移；
  - 不应该泄漏到业务层，让业务围着表结构打转。

---

## 常见问题与注意事项

**Q1：我是不是应该“以表结构作为业务对象”？**  
不应该。  
你现在 domain 层完全看不到表结构，说明你已经在用领域模型抽象业务，这是加分项。

**Q2：仓储一次操作多张表是不是耦合？**  
这是“对聚合负责”的合理耦合，优于 service 手动 orchestrate 多个 DAO 的做法。

**Q3：Service 和 Repository 的边界怎么划？**  
- Repository：围绕聚合的持久化（怎么存/怎么读）；  
- Service：围绕业务流程（什么时候存/什么时候读/存哪些）。

**Q4：Fake 仓储以后还用得上吗？**  
非常用得上：
- 本地快速 demo；
- 单元测试 / 集成测试；
- 做迁移时，用 Fake 把业务跑通，再替换为真实实现。

---

## 最佳实践小结

- 用 **dataclass / pydantic 模型** 描述领域对象，而不是直接暴露 ORM 模型。
- 为每个“聚合”设计一个仓储接口（如 `BasePermissionTable`），而不是为每张表设计 DAO。
- 仓储实现里可以一次操作多张表，只要对外暴露的是**领域模型**，而不是表。
- Service 层做业务编排，不要把 SQL/事务细节都塞进去。
- 用 Fake 仓储支撑本地开发和测试，真实实现再接上 ORM + Alembic。
- 像 `built_in` 这种字段可以先预留，用于未来的“内置数据保护”能力，不影响当前业务。

---

## 小结与下一步

这篇文章我们看到的是：

- 为什么“业务代码里看不到表结构”是正常甚至更好的设计；
- 一个权限组管理模块如何用：
  - 领域模型（PermissionGroup / PermissionGroupDetail）
  - 仓储抽象（BasePermissionTable）
  - Fake 实现 + 真实实现
  来把“业务世界”和“数据库世界”解耦；
- 为什么“一个仓储操作多张表”是**聚合仓储**的合理形式，而不必急着拆给 service。

**如果你正在改造一个现有项目，可以试着这么做：**

1. 先为一个小模块（比如“权限组管理”）画出领域模型；
2. 定义一个仓储接口，只返回/接收领域模型；
3. 写一个 Fake 仓储，让现有 Controller 跑通；
4. 再用 ORM + Alembic 实现一个真实仓储，完全不动上层业务代码。

---

## 参考与延伸阅读

- Eric Evans，《领域驱动设计：软件核心复杂性应对之道》
- Vaughn Vernon，《实现领域驱动设计》
- Martin Fowler: Repository pattern
- 《Clean Architecture》 相关章节：Entities / Use Cases / Gateways / Controllers
- FastAPI 官方文档：关于依赖注入与测试部分  
- SQLAlchemy / Alembic 官方文档：表结构建模与迁移

---

## 行动号召（CTA）

- 可以把这篇文章保存到你的项目 wiki 里，对照着你现有的模块做一轮“表结构 vs 领域模型 vs 仓储”的梳理。
- 如果你已经有一个权限系统，试着先给它画出一个 `PermissionGroupDetail` 这样的聚合，然后看你现在的代码是更像“DAO 拼 Service”，还是“聚合仓储”。
- 有兴趣的话，可以把你现有的权限模块结构贴出来，看看怎么在不大动干戈的情况下，逐步引入这种分层方式。

