---
title: "Alembic 入门：第一次用 SQLAlchemy 做数据库迁移"
date: 2025-11-28
---

# 🐣 Alembic 入门：第一次用 SQLAlchemy 做数据库迁移

## 💡 副标题 / 摘要

如果你已经在用 SQLAlchemy 操作数据库，却还在靠“手工改表结构 + 导出导入 SQL”来维护 schema，这篇文章会带你用最小成本上手 Alembic。  
我们会从 **0 配置 Alembic** 开始，一步步完成：生成迁移、升级/回滚数据库、和 SQLAlchemy 模型联动。

---

## 🎯 目标读者

适合这样的你：

- 已经在项目中使用 **SQLAlchemy**（ORM 或 Core 都行）；  
- 从未使用过 Alembic，或只懂 `alembic upgrade head` 这几个命令；  
- 想为自己的项目加上 **可回滚、可追踪、可审计** 的数据库结构变更；
- 以 Python / Web 后端为主（Flask / FastAPI / 自研框架均可）。

---

## 🔥 背景 / 动机：为什么需要数据库迁移工具？

没有 Alembic 时，我们通常怎么改数据库结构？

- 在本地手改表结构（改字段、加索引）；
- 导出 SQL 发给同事 / DBA；
- 生产环境再手工执行一次；
- 一旦出错，回滚非常痛苦。

常见痛点：

- **多人协作困难**：谁先改？谁后改？改了什么？  
- **环境不一致**：本地、测试、生产的表结构经常不一样；  
- **难以回滚**：一旦上线发现问题，很难安全退回之前版本；  
- **审计困难**：几年后根本不知道这个表为什么多了几个字段。

Alembic 做的事情可以总结为一句话：

> **把“数据库结构的变化”变成一条可回放、可回滚、可审计的时间线。**

---

## 🧩 核心概念：Alembic 里你必须认识的几个词

| 概念                | 说明                                                         |
| ----------------- | ---------------------------------------------------------- |
| **Migration / 迁移** | 一次数据库结构变更（新增表、加字段、删索引等），对应一个 Python 脚本 |
| **Revision / 版本号** | 每个迁移脚本的唯一 ID，通常是一串十六进制字符串                         |
| **Upgrade**       | 从旧版本升级到新版本（执行 `upgrade()` 函数）                     |
| **Downgrade**     | 从新版本回退到旧版本（执行 `downgrade()` 函数）                 |
| **Head**          | 当前迁移链的“最新版本”（头部）                                     |
| **env.py**        | Alembic 的入口文件，负责连接数据库、加载模型、运行迁移                      |
| **versions/**     | 存放所有迁移脚本的目录                                           |

理解这几个词之后，Alembic 就不那么“玄学”，更像是 **git 版本管理的数据库版**：

- `alembic revision` ≈ `git commit`  
- `alembic upgrade` ≈ `git checkout` 到某个提交  
- `alembic history` ≈ `git log`

---

## 🛠 实践指南 / 步骤：第一次用 Alembic 管理你的数据库

假设你现在有一个最小项目结构：

```bash
myapp/
  app.py
  models.py
  db.py
```

### 一、安装 Alembic

在你的虚拟环境中安装：

```bash
pip install alembic
```

验证是否安装成功：

```bash
alembic --version
```

---

### 二、初始化 Alembic 项目

在项目根目录（与 `models.py` 同级）执行：

```bash
cd myapp
alembic init alembic
```

会生成：

```bash
myapp/
  alembic/
    env.py
    script.py.mako
    versions/
  alembic.ini
  app.py
  models.py
  db.py
```

这一步完成了：

- 创建 Alembic 配置文件 `alembic.ini`；  
- 创建存放迁移脚本的目录 `alembic/versions/`；  
- 创建入口 `alembic/env.py`。

---

### 三、配置数据库连接 + 绑定 SQLAlchemy 模型

Alembic 需要知道两件事：

1. **怎么连到数据库**（连接 URL）；  
2. **要对比哪些模型**（`target_metadata`）。

#### 1️⃣ 设置数据库 URL

打开根目录的 `alembic.ini`，找到：

```ini
sqlalchemy.url = driver://user:pass@localhost/dbname
```

改成你项目中用的数据库地址，例如：

```ini
sqlalchemy.url = mysql+pymysql://user:password@127.0.0.1:3306/mydb
```

如果你不想把连接信息写死在 `alembic.ini`，也可以放到环境变量中，然后在 `env.py` 里动态读取（进阶用法，本文先不展开）。

#### 2️⃣ 绑定 `target_metadata`

假设你在 `models.py` 中这样定义模型：

```python
# models.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
```

在 `alembic/env.py` 里引入这个 `Base`，并设置 `target_metadata`：

```python
# alembic/env.py
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from myapp.models import Base  # ← 关键：引入你的 Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata  # ← 关键：告诉 Alembic 你的模型元数据
```

这样，当你使用 `--autogenerate` 时，Alembic 就会拿 `Base.metadata` 里的结构与数据库当前结构做对比。

---

### 四、第一次生成迁移脚本（Autogenerate）

现在数据库中还没有 `users` 这张表，而你的模型里已经定义了它。  
让 Alembic 帮我们生成创建该表的迁移：

```bash
alembic revision --autogenerate -m "create users table"
```

执行后，会在 `alembic/versions/` 下生成一个新文件，例如：

```bash
alembic/versions/
  20251128_123456_create_users_table.py
```

打开这个文件，内容大致是：

```python
from alembic import op
import sqlalchemy as sa

revision = "20251128_123456"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=50), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("users")
```

这里有几点需要理解：

- `upgrade()`：升级时执行，创建 `users` 表；  
- `downgrade()`：回滚时执行，删除 `users` 表；  
- `revision` / `down_revision`：表示“我是谁，我的上一个版本是谁”，用来串成一条迁移链。

**非常重要：每次 autogenerate 生成的脚本，都应该人工 review 一遍，而不是盲目执行。**

---

### 五、应用迁移：升级和回滚数据库

#### 1️⃣ 升级到最新版本（head）

执行：

```bash
alembic upgrade head
```

Alembic 会：

- 连接到你配置的数据库；  
- 在库里创建一个名为 `alembic_version` 的表，记录当前版本号；  
- 执行 `upgrade()`，创建 `users` 表。

此时你可以直接连接数据库，查看表结构是否符合预期。

#### 2️⃣ 回滚到上一个版本

如果你想撤销这次迁移，只要：

```bash
alembic downgrade -1
```

Alembic 会找到“上一个版本”，执行当前脚本的 `downgrade()`，把 `users` 表删掉。

你也可以指定回到某个具体版本：

```bash
alembic downgrade 20251128_123456
```

升级同理：

```bash
alembic upgrade 20251128_123456
```

---

### 六、后续迭代：模型变更 → 迁移脚本 → 升级

后续开发中，你的流程应该尽量变成：

1. 修改 `models.py` 中的模型，比如给 `User` 加一个 `email` 字段：

   ```python
   class User(Base):
       __tablename__ = "users"

       id: Mapped[int] = mapped_column(primary_key=True)
       name: Mapped[str] = mapped_column(String(50))
       email: Mapped[str] = mapped_column(String(100), nullable=True)
   ```

2. 生成迁移脚本：

   ```bash
   alembic revision --autogenerate -m "add email to user"
   ```

   打开生成的脚本，确认内容大致是：

   ```python
   def upgrade() -> None:
       op.add_column("users", sa.Column("email", sa.String(length=100), nullable=True))


   def downgrade() -> None:
       op.drop_column("users", "email")
   ```

3. 执行迁移：

   ```bash
   alembic upgrade head
   ```

4. 在代码中开始使用 `User.email` 字段。

**关键原则：永远让 Alembic 成为“唯一修改数据库结构的入口”。**

---

## 🧪 可运行示例：从零到第一个迁移

下面是一套你可以复制到本地尝试的最小示例。

1. 新建项目目录：

   ```bash
   mkdir alembic-demo
   cd alembic-demo
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   pip install sqlalchemy alembic pymysql
   ```

2. 创建 `models.py`：

   ```python
   # models.py
   from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
   from sqlalchemy import String, Integer


   class Base(DeclarativeBase):
       pass


   class User(Base):
       __tablename__ = "users"

       id: Mapped[int] = mapped_column(Integer, primary_key=True)
       name: Mapped[str] = mapped_column(String(50))
   ```

3. 初始化 Alembic：

   ```bash
   alembic init alembic
   ```

4. 修改 `alembic.ini`：

   ```ini
   sqlalchemy.url = mysql+pymysql://user:password@127.0.0.1:3306/alembic_demo
   ```

  （提前在数据库中建好 `alembic_demo` 这个库。）

5. 修改 `alembic/env.py`，引入 `Base`：

   ```python
   from myproject.models import Base  # 按你的真实包名修改

   target_metadata = Base.metadata
   ```

6. 生成并应用迁移：

   ```bash
   alembic revision --autogenerate -m "create users table"
   alembic upgrade head
   ```

完成后，你的数据库中就会出现 `users` 表和 `alembic_version` 表。

---

## ⚙️ 解释与原理：Alembic 在背后做了什么？

简单理解 Alembic 的内部流程：

1. **版本管理：**
   - 每个迁移脚本都有自己的 `revision` 和 `down_revision`；
   - 数据库里有一张 `alembic_version` 表只存一个字段：当前版本号；
   - 升级时：根据当前版本 → 找到目标版本 → 依序执行 `upgrade()`；
   - 回滚时：按反方向执行 `downgrade()`。

2. **自动对比（autogenerate）是如何工作的：**
   - Alembic 用 `target_metadata` 代表“模型中的结构”；  
   - 连接数据库，读取真实表结构；  
   - 对比两者的差异，生成对应的 `op.create_table` / `op.add_column` 等操作；
   - 把这些操作写入 `versions/*.py`。

3. **为什么必须人工 review 脚本：**
   - 某些类型（如 `Enum`、`server_default`）在不同数据库方言下表现不同；  
   - 未来你可能会加上“数据迁移”逻辑，只靠自动生成不够；  
   - 自动生成不了“业务意图”，例如：给新列填补默认值、迁移旧字段的数据等。

---

## ⚠️ 常见问题与注意事项

| 问题 / 场景                               | 建议做法                                                     |
| -------------------------------------- | ---------------------------------------------------------- |
| `alembic revision --autogenerate` 不生成任何内容 | 检查 `env.py` 是否正确设置 `target_metadata`，以及模型是否真的变更       |
| 生成的脚本与真实期望不一致                       | 手动编辑 `versions/*.py` 中的 `upgrade()` / `downgrade()`     |
| 多人开发时版本号冲突                            | 尽量保持一个人负责一个功能分支的迁移，并及时合并；必要时手工调整 `down_revision` 关系 |
| 想重建一份“干净”的迁移链                         | 在新建数据库环境时可以合并历史迁移；对已有生产环境请慎重，通常只做追加不做重排         |
| 生产环境害怕直接执行迁移                          | 先在测试 / staging 环境完整跑一遍迁移，再上线；必要时导出 SQL 做人工审核           |

---

## 🌟 最佳实践与建议

1. **永远不要直接在数据库里手改结构**，所有变更尽量通过 Alembic 管理。  
2. 每次运行 `--autogenerate` 后，都要打开生成的脚本 **认真看一遍**。  
3. 把 `alembic.ini`、`alembic/`、`versions/` 全部提交到 Git 中，保证团队共享同一套历史。  
4. 在 CI 中加一条“迁移检查”：拉起一个测试库，跑一遍 `alembic upgrade head` 确保脚本可执行。  
5. 对生产数据库执行迁移前，一定要：
   - 有最近的备份；  
   - 在测试环境演练过一次；  
   - 最好有回滚方案（downgrade 或手工 SQL）。

---

## 📚 小结 / 结论

这篇入门文章带你走完了 Alembic 的最小闭环：

- 安装 Alembic，并在项目中初始化；  
- 通过 `env.py` 绑定 SQLAlchemy 模型（`target_metadata`）；  
- 用 `revision --autogenerate` 生成迁移脚本；  
- 用 `upgrade` / `downgrade` 管理数据库版本；  
- 形成“**模型变更 → 生成迁移 → 执行迁移**”的标准流程。

理解了这些，你已经可以在自己的项目里放心使用 Alembic 了。  
后续你还可以继续学习：

- 多环境配置（开发 / 测试 / 生产不同数据库）；  
- 数据迁移、批量更新；  
- 高级干预（`include_object`、`process_revision_directives` 等）—— 可以结合我写的另一篇《如何干预 Alembic：从自动生成到精细控制》一起看。

---

## 🔗 参考与延伸阅读

- Alembic 官方文档：<https://alembic.sqlalchemy.org/>  
- SQLAlchemy 官方文档：<https://docs.sqlalchemy.org/>  
- “Environment & Migration Context”（官方文档中关于 env.py 的章节）  
- 《如何干预 Alembic：从自动生成到精细控制》（同一专栏的进阶篇）

---

## 🏷️ 元信息

- **阅读时长**：8–12 分钟  
- **标签**：`Python`，`Alembic`，`SQLAlchemy`，`数据库迁移`，`后端入门`  
- **SEO 关键词**：Alembic 入门，SQLAlchemy 数据库迁移，alembic tutorial，Python 数据库版本管理  
- **元描述**：这是一篇面向初学者的 Alembic 入门教程，手把手带你从零配置 Alembic，与 SQLAlchemy 模型联动，完成数据库迁移的生成、升级与回滚。

---

## 🚀 行动号召（CTA）

现在就可以在你的项目里试试：

1. 把现有 SQLAlchemy 模型与 Alembic 连接起来；  
2. 用 `alembic revision --autogenerate` 生成第一份迁移脚本；  
3. 在本地新建一个干净数据库，跑一遍 `alembic upgrade head`，感受“从无到有建出全部表”的过程。

如果你在接入 Alembic 的过程中遇到任何问题（配置、命令、脚本冲突等），可以把报错和 `env.py` 片段贴出来，我们可以一条条一起拆。 

