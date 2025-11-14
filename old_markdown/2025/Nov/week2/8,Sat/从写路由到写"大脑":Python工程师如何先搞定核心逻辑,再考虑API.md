# **标题**
从写路由到写“大脑”：Python 工程师如何先搞定核心逻辑，再考虑 API

---

## 副标题 / 摘要

刚入行时，我们常常一上来就写路由、设计接口、想 chat_id / message_id 怎么存，却发现真正的“智力活”——核心逻辑——总是拖到后面。这篇文章带你从「先写接口」的思维，升级到「先写大脑，再接外壳」，并串起来六边形架构、Clean Architecture、DDD 等背后的经典理念。

---

## 目标读者

适合这些同学阅读：

* **1–3 年经验** 的 Python 后端工程师 / AI 应用开发者
* 正在用 **FastAPI / Django / Flask** 等框架写 API 的工程师
* 想从“CRUD 搬砖工”进化为“懂设计、能抽象”的工程师
* 对 **六边形架构 / Clean Architecture / DDD** 有点好奇但没系统看过书的人

---

# 一、背景 / 动机：为什么“先写接口”会卡死自己？

很多刚入行的 Python 工程师（包括你我）会有这样的流程：

1. 产品提一个新需求：做一个 AI 聊天功能。

2. 打开编辑器，第一反应就是：

   * 设计 URL：`POST /api/chat/send_message`
   * 开始写 router：`@app.post("/chat/send")`
   * 想 request body 参数长什么样：`chat_id / message_id / user_id / content`
   * 想数据库表结构：`chats`，`messages`

3. 写了一堆 API、schema、model、迁移脚本之后，才想起来：
   **“那 AI 回复到底是怎么生成的？”**

常见痛点：

* **核心逻辑没有想清楚**：模型怎么调用、prompt 怎么构造、历史记录怎么截断，全是临时拼出来的。
* **逻辑被绑死在框架里**：
  想做一个 CLI 工具快速测试逻辑？发现所有代码都写在路由函数里。
* **改一点东西牵一大堆**：
  想换一个模型 / 调整对话策略，必须改 API 接口代码，甚至影响前端。

你直觉上已经意识到：

> “不管有没有接口，这些功能其实纯后端 / CLI 就可以跑起来，那是不是说明我应该先写核心逻辑？”

答案是：**是，而且这正好踩在一堆软件工程大师的共识上。**

---

# 二、核心概念：这套“先核心后接口”到底叫什么？

这不是某个大师的“绝学”，而是下面这些理念的综合应用：

1. **关注点分离（Separation of Concerns）**

   * 提出者之一：Dijkstra
   * 意思：不同类型的问题（业务逻辑、UI、存储、接口）分开处理。

2. **单一职责原则（SRP）** – Robert C. Martin（Uncle Bob）

   * 一个类 / 模块只应该有一个引起它变化的理由。
   * 一个“既写路由又写模型调用”的函数，就违反了这条。

3. **六边形架构 / 端口与适配器（Hexagonal Architecture / Ports & Adapters）** – Alistair Cockburn

   * 核心领域逻辑在中间，外面是各种适配器：HTTP、CLI、MQ、定时任务……
   * 核心逻辑对“如何对外暴露”不敏感。

4. **整洁架构（Clean Architecture）** – Uncle Bob

   * 内圈：业务规则
   * 外圈：框架、UI、数据库、接口
   * **内圈不能依赖外圈**，反过来可以。

5. **领域驱动设计（DDD）** – Eric Evans

   * 先定义领域模型和领域服务，再考虑 Application / Interface 层。

6. **Unix 哲学**

   * “程序只做好一件事，然后通过组合实现复杂需求。”

我们要做的事，用大白话就是：

> “先写负责‘思考’的那坨代码（大脑），再决定它是被 HTTP 调用，还是被 CLI 调用，还是被定时任务调用。”

---

# 三、实践指南：如何从“先写接口”切换到“先写核心”？

下面我用一个**AI 聊天功能**作为例子，带你从需求到代码走一遍。

## 步骤 1：用一句话描述功能（对自己也要讲清楚）

> “用户输入一段文字，我根据历史对话，用 AI 模型生成一段回复，并保存本轮对话。”

这个简单的小句子，会强迫你把注意力放在**业务本身**，而不是 HTTP 细节。

---

## 步骤 2：先设计“核心函数”，不考虑 HTTP / CLI

这里先写一个**纯 Python 函数/类**，想象它可以被任何方式调用：

```python
# chat_core.py

from typing import List, Tuple

class ChatService:
    def __init__(self, model_client, history_repo):
        self.model_client = model_client
        self.history_repo = history_repo

    def generate_reply(self, user_id: int, chat_id: int, user_message: str) -> str:
        # 1. 拉取历史对话
        history = self.history_repo.load_history(user_id, chat_id)
        # history: List[Tuple[str, str]] -> [(role, content), ...]

        # 2. 组装 prompt
        prompt = self._build_prompt(history, user_message)

        # 3. 调用模型
        raw_reply = self.model_client.generate(prompt)

        # 4. 后处理（截断、过滤等）
        reply = self._post_process(raw_reply)

        # 5. 保存本轮对话
        self.history_repo.save_message(user_id, chat_id, user_message, reply)

        return reply

    def _build_prompt(self, history: List[Tuple[str, str]], user_message: str) -> str:
        # 简化示例：把历史拼成纯文本
        messages = []
        for role, content in history:
            messages.append(f"{role.upper()}: {content}")
        messages.append(f"USER: {user_message}")
        messages.append("ASSISTANT:")
        return "\n".join(messages)

    def _post_process(self, text: str) -> str:
        # 示例：去掉多余空格，限制最大长度
        text = text.strip()
        return text[:2000]
```

注意这里：

* 没有 FastAPI、没有 request、没有 response，什么 HTTP 都没提。
* 只有一个清晰的输入输出：`(user_id, chat_id, user_message) -> reply`。
* `history_repo` 和 `model_client` 也是抽象出来的依赖，可以换实现。

这段代码，就是你的**“领域服务 / 核心逻辑 / 大脑”**。

---

## 步骤 3：写一个 CLI 适配器（证明你逻辑是独立的）

先不用管前端、接口，搞一个命令行工具，自己就能玩：

```python
# cli_chat.py

import argparse
from chat_core import ChatService
from infra.model_client import OpenAIModelClient
from infra.history_repo import InMemoryHistoryRepo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--chat-id", type=int, default=1)
    parser.add_argument("--message", type=str, required=True)
    args = parser.parse_args()

    # 这里先用内存实现，后面再换数据库也行
    model_client = OpenAIModelClient(api_key="YOUR_API_KEY")
    history_repo = InMemoryHistoryRepo()

    service = ChatService(model_client, history_repo)

    reply = service.generate_reply(
        user_id=args.user_id,
        chat_id=args.chat_id,
        user_message=args.message,
    )
    print("AI:", reply)

if __name__ == "__main__":
    main()
```

跑一下：

```bash
python cli_chat.py --message "你好，今天心情有点低落。"
```

如果这一步能跑通，你就已经拥有一个“和 HTTP 完全解耦”的核心聊天逻辑了。

---

## 步骤 4：再把它挂到 HTTP API 上（Framework 只是外壳）

现在才上 FastAPI（或其他框架）：

```python
# api_chat.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from chat_core import ChatService
from infra.model_client import get_model_client
from infra.history_repo import get_history_repo

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: int
    chat_id: int
    message: str

class ChatResponse(BaseModel):
    reply: str

def get_chat_service() -> ChatService:
    return ChatService(
        model_client=get_model_client(),
        history_repo=get_history_repo(),
    )

@router.post("/chat/send", response_model=ChatResponse)
def send_message(req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    reply = service.generate_reply(
        user_id=req.user_id,
        chat_id=req.chat_id,
        user_message=req.message,
    )
    return ChatResponse(reply=reply)
```

你会发现：

* API 层非常薄，只做：

  * 参数解析
  * 调用核心服务
  * 返回结果
* 任何业务上的改动（比如：增加多轮对话压缩）基本都在 `ChatService` 里完成。

---

# 四、可运行示例：最简内存版 AI 聊天（伪模型）

下面给你一个完全可运行、纯本地版的小例子——用一个“假模型”模拟 AI 回复，用内存存聊天记录。

## 文件结构

```text
project/
├── chat_core.py
├── infra.py
├── cli_chat.py
└── api_chat.py
```

## `infra.py`

```python
# infra.py

from typing import List, Tuple, Dict

# 假模型客户端：简单回声 + 固定前缀
class DummyModelClient:
    def generate(self, prompt: str) -> str:
        return "【假模型回复】" + prompt.split("USER:")[-1].split("ASSISTANT:")[0].strip()

# 内存历史记录存储
class InMemoryHistoryRepo:
    def __init__(self):
        # key: (user_id, chat_id) -> List[(role, content)]
        self._store: Dict[tuple, List[Tuple[str, str]]] = {}

    def load_history(self, user_id: int, chat_id: int) -> List[Tuple[str, str]]:
        return self._store.get((user_id, chat_id), [])

    def save_message(self, user_id: int, chat_id: int, user_msg: str, reply: str):
        key = (user_id, chat_id)
        history = self._store.setdefault(key, [])
        history.append(("user", user_msg))
        history.append(("assistant", reply))
```

## `chat_core.py`

```python
# chat_core.py

from typing import List, Tuple

class ChatService:
    def __init__(self, model_client, history_repo):
        self.model_client = model_client
        self.history_repo = history_repo

    def generate_reply(self, user_id: int, chat_id: int, user_message: str) -> str:
        history = self.history_repo.load_history(user_id, chat_id)
        prompt = self._build_prompt(history, user_message)
        raw_reply = self.model_client.generate(prompt)
        reply = self._post_process(raw_reply)
        self.history_repo.save_message(user_id, chat_id, user_message, reply)
        return reply

    def _build_prompt(self, history: List[Tuple[str, str]], user_message: str) -> str:
        messages = []
        for role, content in history:
            messages.append(f"{role.upper()}: {content}")
        messages.append(f"USER: {user_message}")
        messages.append("ASSISTANT:")
        return "\n".join(messages)

    def _post_process(self, text: str) -> str:
        return text.strip()
```

## `cli_chat.py`

```python
# cli_chat.py

import argparse
from chat_core import ChatService
from infra import DummyModelClient, InMemoryHistoryRepo

# 为了简单，这里用单例
_model_client = DummyModelClient()
_history_repo = InMemoryHistoryRepo()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--chat-id", type=int, default=1)
    parser.add_argument("--message", type=str, required=True)
    args = parser.parse_args()

    service = ChatService(_model_client, _history_repo)
    reply = service.generate_reply(args.user_id, args.chat_id, args.message)
    print("AI:", reply)

if __name__ == "__main__":
    main()
```

运行：

```bash
python cli_chat.py --message "你好，我有点好奇六边形架构是啥？"
```

你会看到类似输出：

```text
AI: 【假模型回复】你好，我有点好奇六边形架构是啥？
```

虽然模型是假的，但**架构是真实的**：你已经把“核心逻辑”和“调用方式”分开了。

---

# 五、解释与原理：为什么要这么搞？有什么替代方案？

## 为什么“先写核心逻辑”更靠谱？

1. **可测试性强**

   * 不需要起 HTTP 服务、不需要数据库，就能单元测试核心逻辑。
   * TDD / 单元测试更容易落地。

2. **可复用性高**

   * 一套 ChatService，可以被 HTTP、CLI、WebSocket、公有云函数复用。

3. **降低耦合，降低重构成本**

   * 换模型、加新策略，不动 API 层；
   * 换框架（FastAPI 换成 Django），不动核心逻辑。

4. **团队协作更清晰**

   * 有人专注领域逻辑，有人专注 API 与集成，更容易分工。

## 替代方案 / 其他流派？

* **简单小项目**：有人会说“直接写在路由里就完了”。

  * 对于**一次性小脚本 / demo**，确实可以这么干。
  * 但只要你预感这个功能以后会复杂、有演进，就该一开始就分层。

* **重框架驱动开发**：例如“所有逻辑都是 Django View + ORM”。

  * 好处：上手快、写 CRUD 很爽。
  * 坏处：逻辑被框架锁死，想抽取纯逻辑很费劲。

“先核心后接口”的做法，更偏向**长期投资**，不一定是最“快写完 demo”的，但通常是**最能稳住中长期复杂度**的。

---

# 六、常见问题与注意事项

1. **Q：会不会分层分过头，写一堆 class，显得很重？**

   * 建议：从**最小可拆分单元**开始：

     * 先把“模型调用+prompt 构造+后处理”抽成一个类 / 模块；
     * 日后再慢慢把存储、配置、日志等抽出来。

2. **Q：刚入行同事看不懂这种结构怎么办？**

   * 可以在代码里写一点注释：

     * `# 核心业务逻辑`
     * `# HTTP 适配层`
   * 或在 README 里画一个简单架构图（内圈是 ChatService，外圈是 API/CLI）。

3. **Q：这样会不会影响性能？**

   * 分层本身几乎不带来明显性能损失（多了一两个函数调用而已）。
   * 真正的性能瓶颈多半在 I/O、网络、模型调用上。

4. **Q：安全 / 权限控制放在哪一层？**

   * **认证 / 鉴权**通常放在 API / Application 层；
   * 领域层只在“权限已经被确认”的前提下工作。

---

# 七、最佳实践与建议

给你几点可以直接带走的 checklist：

1. **新功能开发时：先问自己两个问题**

   * “如果没有 HTTP，这个功能能不能作为一个纯 Python 函数存在？”
   * “如果要从命令行调用这功能，我希望的接口长什么样？”

2. **写路由前，先写核心函数 / 核心类**

   * 比如 `ChatService.generate_reply(...)`
   * API 只负责把 HTTP 参数转换成这个函数的参数。

3. **任何时候都警惕“巨型路由函数”**

   * 一旦你发现：路由里有复杂的 if/else、业务判断、模型调用，那就说明该抽出来了。

4. **强迫自己写一个 CLI 或小脚本**

   * 让你从“框架思维”切换到“库思维 / 领域思维”。

5. **记一句话：**

   > “接口是门面，核心逻辑是房子本身。
   > 门面可以重刷，房子结构一旦烂掉，很难重建。”

---

# 八、小结 / 结论：从“写接口”到“写核心”的思维升级

本篇我们做了这些事：

* 从一个真实场景（AI 聊天功能），反思**为什么我们总是先写接口**。
* 串起来了：

  * 关注点分离、单一职责原则
  * 六边形架构 / Clean Architecture
  * DDD、Unix 哲学
* 用一个完整的例子展示了：

  * 核心 `ChatService`
  * CLI 适配器
  * HTTP API 适配器

**下一步你可以做的事情：**

* 把你现有项目里“又大又乱的路由函数”挑一个出来；
* 按文中示例，把“模型调用 + 业务判断”抽成一个 `XXXService`；
* 尝试写一个 CLI 入口直接调用这个 Service，验证你已经分离了核心与接口。

这就是你从“普通 CRUD 后端”向“懂架构的工程师”迈出的一步。

---

# 九、参考与延伸阅读

> *以下是推荐方向，你可以按关键字搜索对应资料：*

* Edsger Dijkstra – *Separation of Concerns*
* Robert C. Martin – *Clean Architecture* / *Agile Software Development, Principles, Patterns, and Practices*
* Alistair Cockburn – *Hexagonal Architecture (Ports & Adapters)*
* Eric Evans – *Domain-Driven Design: Tackling Complexity in the Heart of Software*
* “Unix Philosophy” 相关文章：

  * “Do one thing and do it well”

---

# 十、元信息（Meta 信息）

* **预计阅读时长**：10–15 分钟
* **标签（Tags）**：

  * Python 后端
  * 架构设计
  * 六边形架构
  * Clean Architecture
  * DDD
  * AI 应用开发
* **SEO 关键词（可选）**：

  * Python 核心业务逻辑
  * 六边形架构示例
  * Clean Architecture 实战
  * FastAPI 分层设计
  * AI 聊天服务架构
* **元描述（Meta Description）**：

  > 本文面向 Python 后端与 AI 应用开发者，讲解如何在实现新功能时优先设计核心业务逻辑，再通过六边形架构与 Clean Architecture 的思想，将其暴露为 HTTP API 或 CLI 工具，帮助你从“写接口的工程师”成长为“懂架构的工程师”。

---

# 十一、行动号召（CTA）

* ✍️ **试一试**：
  选你当前项目中的一个接口，把核心逻辑抽出来做成一个 `Service` 类，再补一个 CLI 调用它。

* 🧩 **扩展练习**：
  在这个基础上，再加一个“定时任务”的入口，让同一套核心逻辑支持：API + CLI + 定时任务。

* 💬 **交流与反馈**：
  如果你愿意，可以把你重构前后的代码结构（目录或伪代码）发给我，我可以帮你一起看看还能怎么优化，顺便帮你打磨成一篇对外可发的技术分享或博客。

