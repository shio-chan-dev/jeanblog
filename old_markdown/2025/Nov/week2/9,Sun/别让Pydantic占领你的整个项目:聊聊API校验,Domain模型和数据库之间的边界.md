### 标题

**别让 Pydantic 占领你的整个项目：聊聊 API 校验、Domain 模型和数据库之间的边界**

---

### 副标题 / 摘要

很多用 FastAPI/Pydantic 的 Python 工程师，会不知不觉让 Pydantic Model 贯穿 API、业务、数据库所有层。本文用一个清晰的分层思路和完整代码示例，帮你搞清楚：**Pydantic 适合用在什么地方，Domain / ORM 又应该怎么配合。**

---

### 目标读者

这篇文章适合：

* 正在使用 **FastAPI / Pydantic / SQLAlchemy / SQLModel** 的 Python 后端工程师
* 刚入行 0–3 年、开始关心“分层、架构、领域模型”的开发者
* 想从“会写接口”进阶到“懂业务建模、懂分层”的工程师
* 对 “Pydantic 要不要进 Domain / 要不要用于 DB 模型” 有疑惑的人

---

## 一、背景 / 动机：为什么 Pydantic 容易“长满全项目”？

如果你是从 FastAPI 入门后端，很可能经历过这样的路径：

1. 用 Pydantic 定义请求体、响应体：太好用了，自动校验 + 文档 + 类型提示。
2. 觉得既然 Pydantic 这么香，那干脆：

   * 直接拿 Pydantic Model 当“业务对象”传来传去
   * 甚至顺手拿它去做“数据库模型”

渐渐地，你的项目变成：

* API 层 → Pydantic Model
* Service 层 → Pydantic Model
* DB 层 → 还是 Pydantic Model
* 所有逻辑都在“围绕一个个 BaseModel 子类打转”

短期看起来很爽：

* 少写很多转换代码
* IDE 体验好、自动补全完善

但当项目稍微大一点、复杂一点，你会遇到：

* 想把一些业务逻辑抽出来做脚本 / CLI / 单元测试，却发现强依赖 Pydantic & FastAPI；
* 想换 ORM、换存储，发现“业务层”大量直接依赖某种具体结构；
* Domain 概念跟 API/DB 绑死，业务与框架高度耦合。

这时你就会问出今天这句话：

> **“Pydantic 是不是只应该用在 API 校验？数据库交互能不能不用它？
> Repository 为什么要依赖 Domain 模型，而不是 Pydantic BaseModel？”**

这篇文章，就是来回答：**Pydantic 在一个“分层清晰”的项目中，应该处在什么位置。**

---

## 二、核心概念：我们在说的几种“模型”到底有什么区别？

先把几个关键概念说清楚，不然后面全是名词大战。

### 1. 领域模型（Domain Model）

* 表达的是**业务世界里的真实概念**：文章、用户、订单、库存…
* 不关心 HTTP、JSON、数据库、ORM、Pydantic。
* 可以是 dataclass / 普通类 / NamedTuple 等。

例子：

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

class PostStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"

@dataclass
class Post:
    id: int
    author_id: int
    title: str
    content: str
    status: PostStatus = PostStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None

    def publish(self):
        if self.status == PostStatus.PUBLISHED:
            return
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
```

这里完全没出现 Pydantic。

---

### 2. API 模型 / DTO（Data Transfer Object）

* 目的：**描述请求 & 响应结构**，负责校验 + 文档 + 序列化。
* 典型实现：Pydantic BaseModel。
* 属于 **API 层**，不是业务核心。

例子：

```python
from pydantic import BaseModel
from typing import List

class CreatePostRequest(BaseModel):
    title: str
    content: str
    tags: List[str] = []

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: List[str]
    status: str
```

---

### 3. 持久化模型（Persistence Model / ORM Model）

* 目的：方便和数据库交互（建表、查询、更新、索引）。
* 通常用 ORM：SQLAlchemy / Django ORM / SQLModel / Beanie 等。
* 属于 **Infra / 基础设施层**，和 DB 强相关。

例子（SQLAlchemy）：

```python
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PostTable(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(String, nullable=False)
    status = Column(String(20), default="draft")
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    published_at = Column(DateTime, nullable=True)
```

---

> 🔑 总结一句：
>
> * **Domain Model** = 业务世界
> * **API Model (Pydantic)** = HTTP 世界 / 外部通信
> * **ORM Model** = 数据库世界
>
> 它们可以长得很像，但**职责完全不同**。

---

## 三、实践指南：Pydantic、Domain、DB 的推荐分工（带完整流程）

我们用一个“博客系统”的最小用例来走全流程：

> 用户通过 HTTP 创建一篇文章 → 存到数据库 → 返回文章信息。

我们分 4 层看：

1. API 层：接收 HTTP 请求，用 Pydantic 校验参数
2. Application/Service 层：承载用例逻辑
3. Domain 层：业务真相（Post 实体）
4. Infra/Repo 层：操作数据库（用 ORM）

---

### 步骤 1：定义 Domain 模型（业务核心）

```python
# app/domain/post.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

class PostStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"

@dataclass
class Post:
    id: int
    author_id: int
    title: str
    content: str
    status: PostStatus = PostStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None

    def publish(self):
        if self.status == PostStatus.PUBLISHED:
            return
        self.status = PostStatus.PUBLISHED
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
```

---

### 步骤 2：定义 Repo 接口（用 Domain 类型）

```python
# app/application/ports.py

from typing import Protocol, Optional, List
from app.domain.post import Post

class PostRepository(Protocol):
    def get_by_id(self, post_id: int) -> Optional[Post]: ...
    def save(self, post: Post) -> Post: ...
    def list_published(self, limit: int = 10, offset: int = 0) -> List[Post]: ...
```

> 注意：
>
> * 这里用的是 `Post`（Domain 实体），不是 Pydantic Model
> * 这是 **“我要存的是什么”** 的声明，和 DB 技术无关

---

### 步骤 3：定义 Application Service（用例逻辑）

```python
# app/application/post_service.py

from typing import List
from app.domain.post import Post
from app.application.ports import PostRepository

class PostService:
    def __init__(self, repo: PostRepository):
        self.repo = repo

    def create_draft(self, author_id: int, title: str, content: str, tags: List[str]) -> Post:
        post = Post(
            id=0,  # 具体 ID 由 Repo 决定如何生成
            author_id=author_id,
            title=title,
            content=content,
            tags=tags,
        )
        return self.repo.save(post)
```

Application Service 只关心：

* 拿到 Domain 对象
* 调用 Repo 接口

完全不关心：

* HTTP 怎么传参
* DB 用什么类型字段

---

### 步骤 4：实现 Repo（Infra 层，处理 ORM ↔ Domain 转换）

```python
# app/infra/repositories/postgres_post_repo.py

from typing import Optional, List
from sqlalchemy.orm import Session

from app.domain.post import Post, PostStatus
from app.application.ports import PostRepository
from app.infra.tables import PostTable


class PostgresPostRepository(PostRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, post_id: int) -> Optional[Post]:
        row = self.session.query(PostTable).get(post_id)
        if row is None:
            return None
        return self._row_to_domain(row)

    def save(self, post: Post) -> Post:
        if post.id == 0:
            row = PostTable(
                author_id=post.author_id,
                title=post.title,
                content=post.content,
                status=post.status.value,
                created_at=post.created_at,
                updated_at=post.updated_at,
                published_at=post.published_at,
            )
            self.session.add(row)
        else:
            row = self.session.query(PostTable).get(post.id)
            row.title = post.title
            row.content = post.content
            row.status = post.status.value
            row.updated_at = post.updated_at
            row.published_at = post.published_at

        self.session.commit()
        self.session.refresh(row)
        return self._row_to_domain(row)

    def list_published(self, limit: int = 10, offset: int = 0) -> List[Post]:
        q = (
            self.session.query(PostTable)
            .filter(PostTable.status == PostStatus.PUBLISHED.value)
            .order_by(PostTable.published_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [self._row_to_domain(row) for row in q.all()]

    def _row_to_domain(self, row: PostTable) -> Post:
        return Post(
            id=row.id,
            author_id=row.author_id,
            title=row.title,
            content=row.content,
            status=PostStatus(row.status),
            tags=row.tags or [],
            created_at=row.created_at,
            updated_at=row.updated_at,
            published_at=row.published_at,
        )
```

> 这里是 DB 的“重灾区”，但你会看到：
>
> * **Repo 实现依赖 Domain 实体是正常且推荐的**
> * 上层完全不需要关心 ORM 的存在

---

### 步骤 5：API 层才用 Pydantic（请求校验 + 响应包装）

```python
# app/api/schemas.py

from pydantic import BaseModel
from typing import List

class CreatePostRequest(BaseModel):
    title: str
    content: str
    tags: List[str] = []

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: List[str]
    status: str
```

```python
# app/api/routes_posts.py

from fastapi import APIRouter, Depends
from app.api.schemas import CreatePostRequest, PostResponse
from app.application.post_service import PostService

router = APIRouter()

def get_post_service() -> PostService:
    # 这里注入具体的 Repo 实现
    ...

@router.post("/posts", response_model=PostResponse)
def create_post(
    req: CreatePostRequest,
    service: PostService = Depends(get_post_service),
):
    # 这里可以从 token 里拿 author_id，这里简化写死
    post = service.create_draft(
        author_id=1,
        title=req.title,
        content=req.content,
        tags=req.tags,
    )
    return PostResponse(
        id=post.id,
        title=post.title,
        content=post.content,
        tags=post.tags,
        status=post.status.value,
    )
```

到这里，你就完成了一个**分工清晰**的流：

* Pydantic 只在 API 层出现
* Domain 是纯 Python，可在 CLI/脚本/别的服务里复用
* DB 相关的都在 Infra/Repo 实现里

---

## 四、解释与原理：为什么要这么分？替代方案有哪些？

### 1. 为什么推荐 Pydantic 只在 API/外围使用？

核心原因：**“依赖方向” & “业务核心解耦框架”**

* Pydantic 本质是一个“工具库 + 框架依赖”（尤其在 FastAPI 中）
* 如果 Domain / Service 直接依赖 Pydantic：

  * 你的业务逻辑就被强绑定在这个库上
  * 换框架 / 去掉 Pydantic / 做纯脚本时，会非常痛苦

而我们更希望：

* 业务核心只依赖 Python 标准库 / 基本类型
* 框架 & 库是**可以替换的外层**，而不是“镶嵌进业务里”的

这其实就是：

* Clean Architecture / 六边形架构 / DDD 里说的：

  * **“内圈（业务）不依赖外圈（框架/技术细节）”**

---

### 2. 替代方案 & 工程妥协

现实工程里，你会看到几种常见做法：

#### 做法 A：全项目统一用 Pydantic Model（不推荐做大项目）

* 最简单、最少代码、最适合 demo & 小玩具。
* 一旦项目复杂度提升，很难控制边界。

#### 做法 B：SQLModel / Beanie 等“Pydantic + ORM 一体化”

* 对小中型项目其实挺香：

  * 一份模型同时用于 API & DB
* 但如果你想走比较“纯粹”的领域建模路线：

  * 建议仍然在 Domain 层用独立实体，
  * 把 SQLModel 当成 Infra 的实现。

#### 做法 C：本文推荐方案（分层明确）

* Domain：纯 Python 实体
* API：Pydantic DTO
* DB：ORM Model
* Repo：负责做转换

适合：你希望后面养成“架构感”，不只写 CRUD 的情况。

---

## 五、常见问题与注意事项

### Q1：这样要写很多“转换代码”，是不是很麻烦？

是的，会多一点代码，但换来的是：

* **职责清晰**：哪里是业务，哪里是通信，哪里是存储，一目了然；
* **改动可控**：换 ORM、换框架，不会牵一发动全身；
* **测试更简单**：Domain & Service 层可以完全不依赖 FastAPI 进行单元测试。

小建议：

* 用一些小工具函数封装：`post_to_response(post)`、`row_to_post(row)` 等；
* 真正麻烦的不是“写转换”，而是“到处混在一起然后不知道怎么改”。

---

### Q2：Repo 接口依赖 Domain 实体，会不会算“反向依赖”？

不会，反而是应该的：

* Repo 的职责就是：“存取领域对象”；
* 所以它非常自然地要用 Domain 类型；
* **错的做法**是：Domain 反过来依赖具体 Repo 实现（比如直接 import SQLAlchemy Session）。

---

### Q3：是不是 DB 层“绝对不能”用 Pydantic？

不是“绝对不能”，而是：

* **不要让 Pydantic Model 渗透进 Domain & Service 层**
* 在 Infra 里，用 Pydantic 做一些验证/转换是完全可以的
* 比如：

  * 用 Pydantic 校验某个外部系统的配置
  * 用 Pydantic 解析第三方 API 响应，再转成 Domain

---

## 六、最佳实践与建议（可以当 Checklist）

1. **Pydantic 放在哪：**

   * ✅ API 层请求/响应 DTO
   * ✅ 配置 / 外部服务数据结构
   * ❌ 不要作为 Domain 实体
   * ❌ 不要作为 Repo 接口类型

2. **Domain 模型如何写：**

   * 用 dataclass / 普通类
   * 集中业务规则（状态变更、校验）
   * 不 import FastAPI / Pydantic / ORM

3. **Repo 的职责：**

   * 接收 & 返回 Domain 实体
   * 在实现中负责 ORM/Pydantic ↔ Domain 之间的转换
   * 不把 ORM/Pydantic 透传给上层

4. **改造老项目的顺序：**

   * 先从最核心的一两个领域对象开始抽出 Domain 实体
   * 再在 Service 层用 Domain，Repo 层慢慢替换
   * API 层最后做 Pydantic ↔ Domain 的转换

---

## 七、小结 / 结论：一句话记住这件事

> **Pydantic 是用来“和外界打交道”的，不是用来“定义你业务世界本身”的。**
>
> * API / 配置 / 外部服务 → 用 Pydantic 很合适
> * 业务核心（Domain） → 尽量保持纯 Python
> * 数据库交互 → 用 ORM / SQL，Repo 负责 Domain ↔ DB 的翻译

当你开始有意识地把这三者分开，你会发现：

* 代码更容易测、更容易重构、更容易解释给别人听；
* 你不再只是“写接口的人”，而是在用代码表达你的业务理解。

---

## 八、参考与延伸阅读（建议你按关键词搜索）

> *避免直接贴长链接，你可以按这些关键字搜索官方文档/博客：*

* “FastAPI Pydantic Models” – FastAPI 官方文档
* “Pydantic Usage Models vs ORM” – Pydantic 文档讨论
* “Repository Pattern in Python”
* “Clean Architecture in Python”
* “Domain-Driven Design (DDD) Domain Model”

---

## 九、元信息（Meta 信息）

* **预计阅读时长**：12–18 分钟
* **标签（Tags）**：

  * FastAPI
  * Pydantic
  * Python 后端
  * 分层架构
  * Domain Model
  * Repository Pattern
* **SEO 关键词（Keywords）**：

  * Pydantic 只用于 API 校验
  * FastAPI 分层设计
  * Python Domain Model 与 Pydantic
  * Repository 依赖 Domain 实体
  * Pydantic 与 SQLAlchemy 分层实践
* **元描述（Meta Description）**：

  > 本文面向使用 FastAPI/Pydantic 的 Python 后端工程师，深入讲解 Pydantic 在分层架构中的正确位置：如何只在 API 层使用 Pydantic 做校验与序列化，把 Domain 模型和数据库交互从框架中解耦，并通过完整示例展示 Repository 与 Domain 的协作方式。

---

## 十、行动号召（CTA）

* 🛠 **动手试一试**：
  从你现有项目中挑一个核心实体（比如 `User` 或 `Post`），
  把它从 Pydantic Model 中独立出来，改成一个纯 Python 的 Domain 类，
  然后让 Service / Repo 都改用这个 Domain 类。

* 🧪 **写个小实验仓库**：
  新建一个极简 FastAPI + SQLAlchemy 项目，按本文结构搭一遍分层，以后新项目可以直接 copy 这套脚手架。


