# 🧱 从零到生产：如何优雅地设计 ORM 层管理（以 SQLAlchemy 为核心）

> 本文将带你从数据库表结构出发，构建一套高内聚、低耦合的 ORM 层架构。
> 目标：让你的 Flask / FastAPI 项目在数据访问上既简洁又稳健。

---

## 一、为什么要重视 ORM 层设计？

很多项目初期只是“先能跑”，直接把 SQL 写在控制器里，但很快就会出现：

* 业务逻辑和 SQL 混在一起；
* 表关系复杂，维护困难；
* 想复用查询逻辑很麻烦；
* 迁移到别的框架（Flask → FastAPI）代价大。

ORM 层（Object Relational Mapping）是数据库与业务逻辑之间的 **抽象桥梁**，
一个好的 ORM 层能让你只关心对象，不用反复写 SQL。

---

## 二、项目场景：招标信息数据系统

我们以一个真实业务为例：
爬取各网站的招标公告，保存为结构化数据，并生成统计看板。

### 目标数据库实体

| 表名                    | 功能         |
| --------------------- | ---------- |
| `tender_info`         | 公告基本信息     |
| `tender_attachments`  | 公告及变更文件    |
| `tender_organization` | 招标机构与联系方式  |
| `tender_statistics`   | 每日/月/年统计信息 |

---

## 三、ORM 层设计思路

### 🧩 分层原则

| 层级               | 作用                | 代码位置               |
| ---------------- | ----------------- | ------------------ |
| **Model 层**      | ORM 模型定义，对应数据库表结构 | `models.py`        |
| **Repository 层** | 封装 CRUD 逻辑（数据库操作） | `repository.py`    |
| **Service 层**    | 业务逻辑层（聚合多个仓库逻辑）   | `service.py`       |
| **API 层**        | 控制器/路由接口          | Flask/FastAPI 视图文件 |

这种分层让你做到：

* 一处改模型，多处复用；
* 业务与数据库访问解耦；
* ORM 模型可被多框架复用。

---

## 四、ORM 模型定义（SQLAlchemy 2.x）

我们使用 `declarative_base()` 定义所有模型类。
四张表如下：

```python
# models.py
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Date, DateTime, Text, Enum, JSON,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()
```

---

### 1️⃣ 基本信息表 `TenderInfo`

```python
class TenderInfo(Base):
    __tablename__ = "tender_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tender_id = Column(String(100), nullable=False, index=True, comment="项目编号/招标编号")
    tender_title = Column(String(500), nullable=False, comment="公告标题")
    announcement_type = Column(String(255), comment="公告类型")
    purchase_type = Column(String(100), comment="采购方式")
    tender_status = Column(String(100), comment="项目状态")
    website_source = Column(String(50), comment="网站来源")
    announcement_date = Column(Date, comment="公告日期")
    bid_doc_deadline = Column(String(100))
    bid_open_time = Column(String(100))
    has_change_announce = Column(Enum("Y", "N", name="change_enum"), default="N")
    change_content = Column(Text)
    winning_bidder = Column(String(255))
    collection_time = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 一对多：附件
    attachments = relationship(
        "TenderAttachments",
        back_populates="tender",
        cascade="all, delete-orphan",
        foreign_keys="TenderAttachments.tender_info_id"
    )

    # 一对一：机构信息
    organization = relationship(
        "TenderOrganization",
        back_populates="tender",
        uselist=False,
        foreign_keys="TenderOrganization.tender_info_id"
    )

    def __repr__(self):
        return f"<TenderInfo(id={self.id}, title='{self.tender_title}', source='{self.website_source}')>"
```

---

### 2️⃣ 附件信息表 `TenderAttachments`

```python
class TenderAttachments(Base):
    __tablename__ = "tender_attachments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tender_info_id = Column(
        Integer, ForeignKey("tender_info.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    original_announcement_url = Column(Text)
    original_announcement_file_path = Column(Text)
    files_url = Column(String(500))
    change_announcement_url = Column(Text)
    change_announcement_file_path = Column(String(512))
    change_files_url = Column(String(500))
    has_attachments = Column(Enum("Y", "N", name="attach_enum"), default="N")
    created_at = Column(DateTime, default=datetime.utcnow)

    tender = relationship("TenderInfo", back_populates="attachments")
```

---

### 3️⃣ 招标机构表 `TenderOrganization`

```python
class TenderOrganization(Base):
    __tablename__ = "tender_organization"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tender_info_id = Column(
        Integer, ForeignKey("tender_info.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True
    )
    purchaser = Column(String(200))
    tender_agency = Column(String(255))
    contact_person = Column(String(100))
    contact_phone = Column(String(50))
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    tender = relationship("TenderInfo", back_populates="organization")
```

---

### 4️⃣ 统计表 `TenderStatistics`

```python
class TenderStatistics(Base):
    __tablename__ = "tender_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_date = Column(Date, nullable=False)
    period_type = Column(String(10), nullable=False)  # daily / monthly / yearly
    website_source = Column(String(20), default="all")
    total_count = Column(Integer, default=0)
    cumulative_total = Column(Integer, default=0)
    announcement_type_stats = Column(JSON)
    purchase_type_stats = Column(JSON)
    tender_status_stats = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 五、Repository 层（数据访问层）

这一层的职责是封装具体的数据库操作逻辑，保证 API 或 Service 层不直接访问 Session。

```python
# repository.py
from sqlalchemy.orm import Session
from models import TenderInfo, TenderAttachments, TenderOrganization, TenderStatistics

def create_tender_info(db: Session, data: dict):
    tender = TenderInfo(**data)
    db.add(tender)
    db.commit()
    db.refresh(tender)
    return tender

def get_tender_info(db: Session, tender_id: str):
    return db.query(TenderInfo).filter(TenderInfo.tender_id == tender_id).first()

def list_tenders(db: Session, limit=50):
    return db.query(TenderInfo).order_by(TenderInfo.collection_time.desc()).limit(limit).all()

def create_attachments(db: Session, tender_info_id: int, data: dict):
    attachment = TenderAttachments(tender_info_id=tender_info_id, **data)
    db.add(attachment)
    db.commit()
    return attachment

def create_organization(db: Session, tender_info_id: int, data: dict):
    org = TenderOrganization(tender_info_id=tender_info_id, **data)
    db.add(org)
    db.commit()
    return org

def insert_statistics(db: Session, data: dict):
    stat = TenderStatistics(**data)
    db.add(stat)
    db.commit()
    return stat
```

---

## 六、Service 层（业务逻辑聚合）

Service 层负责“协调多个仓库操作”，
让控制器不直接操作数据库。

```python
# service.py
from sqlalchemy.orm import Session
from repository import create_tender_info, create_attachments, create_organization

def create_full_tender(db: Session, info_data, attachment_data, org_data):
    tender = create_tender_info(db, info_data)
    create_attachments(db, tender.id, attachment_data)
    create_organization(db, tender.id, org_data)
    return tender
```

---

## 七、API 层（Flask / FastAPI 通用）

### Flask 示例

```python
# app.py
from flask import Flask, jsonify, request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from repository import list_tenders, get_tender_info

app = Flask(__name__)
engine = create_engine("sqlite:///tenders.db")
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

@app.route("/api/tenders")
def get_all_tenders():
    with SessionLocal() as db:
        tenders = list_tenders(db)
        return jsonify([{"id": t.id, "title": t.tender_title} for t in tenders])

@app.route("/api/tenders/<tid>")
def get_one_tender(tid):
    with SessionLocal() as db:
        t = get_tender_info(db, tid)
        return jsonify({
            "id": t.id,
            "title": t.tender_title,
            "status": t.tender_status,
            "source": t.website_source
        })
```

---

## 八、ORM 管理层的核心思想

| 原则        | 说明                          |
| --------- | --------------------------- |
| **职责单一**  | ORM 只映射对象，不混入业务逻辑           |
| **解耦层次**  | CRUD 放在 Repository 层        |
| **聚合操作**  | 复杂逻辑放 Service 层             |
| **自动关系**  | 充分利用 relationship 代替手写 JOIN |
| **可扩展性强** | 新表可独立添加，不影响旧层逻辑             |

---

## 九、ORM 设计的最佳实践

✅ **推荐做法**

* 为每个模型定义 `__repr__`，便于调试
* 在外键列加索引（`index=True`）
* 在一对一外键上加唯一约束（`unique=True`）
* 使用 `back_populates` 保持双向同步
* 使用 `cascade="all, delete-orphan"` 自动级联删除

🚫 **不要做的事**

* 不要在 API 层直接使用 `Session`
* 不要让模型类承担业务逻辑
* 不要在模型类里定义复杂查询方法（放 Repository 层）

---

## 🔚 十、总结

你现在拥有了一整套可扩展的 ORM 管理结构：

```
📦 your_project/
 ┣━ models.py           # ORM 模型定义
 ┣━ repository.py       # 数据访问层
 ┣━ service.py          # 业务聚合层
 ┣━ app.py              # Flask / FastAPI 路由
 ┗━ database.py         # Engine + Session 配置
```

优点：

* 框架无关（Flask / FastAPI 均可）
* ORM 与业务逻辑解耦
* 表关系清晰，一对多/一对一自然可读
* 扩展性极强（加表无需重构）

---
