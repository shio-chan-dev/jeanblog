---
title: "如何干预 Alembic：从自动生成到精细控制"
date: 2025-11-28
---

# 🧬 如何干预 Alembic：从自动生成到精细控制

## 💡 副标题 / 摘要

大多数人用 Alembic 的方式是：改 SQLAlchemy 模型 → `alembic revision --autogenerate` → `alembic upgrade head`。  
但在真实项目里，你往往需要“插手”这条流水线：控制生成的迁移内容、插入数据迁移、在生产环境加保护、按分支管理多套 Schema……

这篇文章会带你系统认识 **“如何干预 Alembic”**：  
从 `env.py` 到单个迁移脚本，从自动生成到手写数据迁移，让你能放心地在生产库上使用 Alembic，而不是被它“牵着走”。

---

## 🎯 目标读者

适合以下读者：

- 已在项目中使用 **SQLAlchemy + Alembic**；
- 希望从“只会用 autogenerate”进阶到“懂得控制 Alembic 行为”；
- 有生产库 / 多环境（dev、staging、prod）场景，需要更安全的迁移控制；
- 想把 **数据迁移**、**自定义检查**、**安全保护** 加进 Alembic 流程的后端工程师。

---

## 🔥 背景 / 动机：为什么要“干预” Alembic？

只使用 Alembic 的默认玩法，很容易遇到这些问题：

- `--autogenerate` 生成了一堆你不理解的操作，不敢在生产上跑；
- 模型删了字段，自动生成的迁移脚本也直接删列，但生产上其实还有老数据需要兜底；
- 想在迁移时顺便初始化一些字典表、配置表，但不知放在哪；
- 有些表只在测试 / demo 环境需要，生产环境不想创建；
- 多个服务共享一个数据库，需要 **按分支/模块控制迁移范围**。

要解决这些问题，你就必须学会：  
**在 Alembic 的各个“接缝处”插入自己的逻辑**。

---

## 🧩 核心概念：Alembic 里可“动手脚”的关键点

| 概念 / 位置                         | 作用 / 可干预点                                       |
| ---------------------------------- | --------------------------------------------------- |
| `env.py`                           | Alembic 的入口文件，控制如何连接 DB、如何运行迁移、如何生成版本脚本 |
| `target_metadata`                  | 通常指向 SQLAlchemy 的 Base.metadata，用于 autogenerate 对比 |
| 迁移脚本 `versions/*.py`           | 每个 revision 对应一个文件，包含 `upgrade()` / `downgrade()` 逻辑 |
| `op` 对象 (`alembic.op`)           | 在迁移脚本中用于执行 schema / data 修改的操作集合             |
| `process_revision_directives` 钩子 | 在 Autogenerate 产生 revision 时，允许你修改 / 丢弃生成结果        |
| `include_object` 回调              | 控制哪些表 / 列会参与 autogenerate 对比                       |
| `offline` / `online` 模式          | 控制是生成 SQL 文件，还是直接连数据库执行                            |

理解这些点，就知道该从哪几个地方“插手” Alembic 了。

---

## 🛠 实践指南：一步步在 Alembic 流程中“插手”

下面假设你已经有一个标准的 Alembic 项目结构（使用 SQLAlchemy 2.x / 1.4）：

```bash
alembic init alembic
```

目录大致如下：

```bash
alembic/
  env.py
  script.py.mako
  versions/
alembic.ini
```

### 一、在 env.py 里插入你的规则

`env.py` 是 Alembic 的“大总管”，我们最常做的三类干预：

1. **绑定 SQLAlchemy 的元数据**，让 autogenerate 只对比你想管的模型；
2. **过滤对象**，例如跳过某些表或某些列；
3. **在生成 revision 时二次检查 / 修改内容**。

假设你有一个 `models.py`，其中定义了 SQLAlchemy 的 `Base`：

```python
# models.py
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    ...
```

在 `env.py` 中引入它，并配置 `target_metadata`：

```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from myproject.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
```

#### 1️⃣ 过滤不需要迁移的表 / 列：`include_object`

例如，你不想让 Alembic 管理一些日志表、临时表：

```python
def include_object(object, name, type_, reflected, compare_to):
    # 跳过以 tmp_ 开头的临时表
    if type_ == "table" and name.startswith("tmp_"):
        return False

    # 跳过以 _bak 结尾的备份表
    if type_ == "table" and name.endswith("_bak"):
        return False

    return True
```

在 `run_migrations_online` 中把它挂上去：

```python
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,   # 类型变化也参与对比
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()
```

**这样做的好处：**

- 一些辅助/日志/备份表不会出现在 autogenerate 的 diff 里；
- 你可以把“真正的业务表”当成版本控制的唯一来源。

#### 2️⃣ 拦截 autogenerate 结果：`process_revision_directives`

当你执行：

```bash
alembic revision --autogenerate -m "add user status"
```

Alembic 会生成一个 revision 文件。  
在生成前后，你可以用 `process_revision_directives` 进行“二次加工”：

```python
from alembic.operations import ops


def process_revision_directives(context, revision, directives):
    script = directives[0]

    # 没有任何变更时，阻止生成空的迁移文件
    if script.upgrade_ops.is_empty():
        raise SystemExit("No changes in schema detected.")

    # 示例：如果检测到对关键表的删除，就强制失败，要求人工确认
    for op in script.upgrade_ops.ops:
        if isinstance(op, ops.DropTableOp) and op.table_name == "users":
            raise SystemExit("Danger: attempt to drop 'users' table in autogenerate.")
```

在 `env.py` 中挂上：

```python
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    process_revision_directives=process_revision_directives,
    ...
)
```

这就是对 autogenerate “插手”的经典姿势：

- 没有 diff 就拒绝生成空迁移；
- 对敏感表 / 操作施加额外保护；
- 甚至可以根据规则拆分成多个 revision（高级玩法）。

---

### 二、在迁移脚本中插入“数据迁移”逻辑

很多人以为 Alembic 只能做表结构迁移。  
事实上，只要你需要，你完全可以在 `upgrade()` / `downgrade()` 中写 **数据迁移**。

一个典型场景：给 `users` 表新增 `status` 字段，并根据历史数据填充：

```bash
alembic revision --autogenerate -m "add user status"
```

生成的迁移脚本大致会长这样（简化版）：

```python
from alembic import op
import sqlalchemy as sa

revision = "202511280001_add_user_status"
down_revision = "202511270001_prev"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("status", sa.String(length=20), nullable=True))

    # 在此处插入数据迁移逻辑
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE users SET status = :default_status WHERE status IS NULL"
        ),
        {"default_status": "active"},
    )

    # 如果你希望最后变为非空，可以再执行一次 ALTER
    op.alter_column("users", "status", existing_type=sa.String(length=20), nullable=False)


def downgrade() -> None:
    op.drop_column("users", "status")
```

**注意几点：**

- 使用 `op.get_bind()` 获取当前连接，而不是新建 engine；
- 尽量使用 `sa.text` 或 ORM 层的语句，而不是拼接字符串 SQL；
- 大批量数据迁移要评估锁时间和事务大小，可以拆批次执行或线下预处理。

---

### 三、根据环境干预：开发 / 测试 / 生产差异

有些迁移逻辑只想在开发环境运行，例如：

- 初始化 demo 数据；
- 创建测试用的 mock 表；
- 填充只有本地需要的配置。

你可以在 `env.py` 中读取环境变量，例如：

```python
import os

ENV = os.getenv("ALEMBIC_ENV", "dev")
```

然后在 `context.configure` 中传入：

```python
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    render_as_batch=True,
    user_defined={"env": ENV},
)
```

在迁移脚本中读取：

```python
from alembic import op


def upgrade() -> None:
    context = op.get_context()
    env = context.opts.get("env", "dev")

    if env == "prod":
        # 生产环境跳过 demo 数据初始化
        return

    # 开发 / 测试环境执行 demo 数据插入
    conn = op.get_bind()
    conn.execute(...插入一些样例数据...)
```

这样你就可以在同一份迁移脚本中，根据运行环境有选择地执行逻辑。

---

### 四、让 Alembic 和 SQLAlchemy 模型保持“健康关系”

很多项目里，Alembic 和 SQLAlchemy 的关系是这样的：

- 模型改了，但没有更新迁移脚本 → 环境不一致；
- 或者直接在数据库里手改了表结构 → autogenerate 看到一堆脏 diff。

更合理的姿势是：

1. **只允许通过 Alembic 修改数据库结构**；
2. 每次模型变更后，第一时间生成并 review 迁移脚本；
3. 在 CI 中增加一个“schema drift 检查”：

   - 利用 Alembic 的 autogenerate 模式生成一个临时 diff；  
   - 如果发现 diff 非空，就认为存在未提交的迁移。

伪代码示意：

```bash
alembic revision --autogenerate -m "check drift" --rev-id tmp_check --head head --splice
# 脚本生成后，检测是否有内容，如果有则 fail
```

实际项目中你可以用脚本分析 `versions/` 是否出现新的文件来做自动化检查。

---

## 🧪 可运行示例：一个“可干预”的 env.py 雏形

下面是一个简化后的 `env.py` 片段，组合了前面提到的几个关键点（过滤对象 + 处理 autogenerate + 传入环境信息）：

```python
import os
from logging.config import fileConfig

from alembic import context
from alembic.operations import ops
from sqlalchemy import engine_from_config, pool

from myproject.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
ENV = os.getenv("ALEMBIC_ENV", "dev")


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and name.startswith("tmp_"):
        return False
    return True


def process_revision_directives(context, revision, directives):
    script = directives[0]

    if script.upgrade_ops.is_empty():
        raise SystemExit("No schema changes detected, cancel revision.")

    for op_ in script.upgrade_ops.ops:
        if isinstance(op_, ops.DropTableOp) and op_.table_name == "users":
            raise SystemExit("Refuse to drop 'users' table automatically.")


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            process_revision_directives=process_revision_directives,
            compare_type=True,
            user_defined={"env": ENV},
        )

        with context.begin_transaction():
            context.run_migrations()
```

把这个思路移植到你的项目，就已经迈出了“干预 Alembic”实践的第一步。

---

## ⚙️ 解释与原理：Alembic 是怎么“跑”起来的？

理解 Alembic 的工作方式，有助于你知道能从哪几层下手：

- 所有 Alembic 命令最终都会调用 `env.py` 中的 `run_migrations_offline` / `run_migrations_online`；
- `context.configure(...)` 相当于告诉 Alembic：**我要迁移哪个 DB、对比哪些元数据、用哪些回调**；
- `context.run_migrations()` 内部会：
  - 确定当前数据库的 revision；
  - 根据要升级/降级到的目标 revision 计算出路径；
  - 依次导入 `versions/` 目录里的脚本，调用其中的 `upgrade()` / `downgrade()`；
- 在 `revision --autogenerate` 时：
  - Alembic 会拿 `target_metadata` 与数据库真实结构对比；
  - 生成一组“操作”（`UpgradeOps` / `DowngradeOps`）；
  - 调用 `process_revision_directives`，给你最后一次修改/拦截这些操作的机会；
  - 再基于模板生成脚本文件。

**一句话概括：**  
`env.py` 负责“调度和规则”，`versions/*.py` 负责“具体动作”，你可以在这两层插手几乎所有关键行为。

---

## ⚠️ 常见问题与注意事项

| 问题 / 场景                                 | 建议做法                                                     |
| ---------------------------------------- | ---------------------------------------------------------- |
| autogenerate 生成了奇怪的 diff（特别是 enum、default） | 关闭对应的 compare 选项，或在 `include_object` / `process_revision_directives` 中过滤 |
| 迁移脚本里写了复杂数据迁移导致超时 / 死锁               | 尽量拆成多次小批量更新；考虑先线下迁移数据，再在 Alembic 中只做 schema 变更       |
| 生产环境不小心执行了错误迁移                       | 启用备份&回滚策略；确保在 CI 中跑完迁移测试，再在生产部署前人工 review            |
| 多服务共享一个数据库，迁移时互相影响                   | 使用 `branch_labels` 划分迁移分支，或为不同服务使用不同的 `versions` 目录          |
| 想“重置”所有版本，从头来过                        | 谨慎操作：通常是在新建空库时重建迁移链，不建议在已有生产数据的库上硬重置            |

---

## 🌟 最佳实践与建议

1. **永远 review 自动生成的迁移脚本**，不要直接在生产上运行未经 review 的 `--autogenerate` 结果。
2. 在 `env.py` 中配置好：
   - `target_metadata`；
   - `include_object`；
   - `process_revision_directives`；
   让 Alembic 只关注你真正关心的对象。
3. 把 “数据迁移” 和 “schema 迁移” 分开思考：  
   如果数据量巨大，考虑脚本化分批迁移，而不是全部塞进 Alembic。
4. 在 CI 中加入一项检查：确保模型与数据库 schema 没有“漂移”（未提交的变更）。
5. 对生产环境执行迁移前，至少做到：
   - 有备份；
   - 有 dry-run / staging 演练；
   - 有清晰的回滚路径。

---

## 📚 小结 / 结论

这篇文章带你从三个层面理解“如何干预 Alembic”：

- 在 `env.py` 层面：通过 `include_object`、`process_revision_directives`、`user_defined` 等机制，控制 **生成哪些迁移、如何生成、在什么环境下运行**；
- 在单个迁移脚本层面：通过 `op.get_bind()` + SQL / ORM 语句实现 **安全的数据迁移**；
- 在工程实践层面：通过 CI 检查、环境分离、Review 习惯，让 Alembic 成为你团队的基础设施，而不是风险来源。

如果你已经在项目中使用 Alembic，建议从一件小事开始实践干预：

- 先给 `env.py` 加上 `process_revision_directives`，拒绝生成“空迁移”和“危险迁移”。  
等你熟悉之后，再逐步把数据迁移、多环境控制等能力叠加上去。

---

## 🔗 参考与延伸阅读

- Alembic 官方文档：<https://alembic.sqlalchemy.org/>
- SQLAlchemy 官方文档：<https://docs.sqlalchemy.org/>
- 关于 autogenerate 的官方说明（Environment & Migration Context 章节）
- 一些大型项目的迁移实践分享（可搜索：Alembic migration best practices）

---

## 🏷️ 元信息

- **阅读时长**：10–15 分钟
- **标签**：`Python`，`Alembic`，`SQLAlchemy`，`数据库迁移`，`后端工程实践`
- **SEO 关键词**：Alembic 干预，Alembic env.py，SQLAlchemy 数据库迁移，autogenerate 最佳实践
- **元描述**：本文系统介绍如何在 Alembic 中“插手”迁移流程，从 env.py 配置、autogenerate 干预到数据迁移与多环境控制，帮助后端工程师在生产环境安全地使用 Alembic。

---

## 🚀 行动号召（CTA）

现在就回到你的项目里，做下面三件小事：

1. 在 `env.py` 中引入 `target_metadata`，确保 Alembic 只对比你真实维护的模型；
2. 加上一个简单的 `process_revision_directives`，阻止空迁移和危险操作；
3. 找一个真实的字段变更需求，用“结构迁移 + 数据迁移”配合完成一次完整的 Alembic 干预练习。

如果你愿意，可以把你的 `env.py` 配置或有趣的迁移坑发出来，一起交流如何把 Alembic 用得更稳、更优雅。

