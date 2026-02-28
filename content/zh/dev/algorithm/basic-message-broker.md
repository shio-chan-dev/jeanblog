---
title: "手写一个基础消息代理：发布、订阅、重试与失败契约"
subtitle: "从最小可用到工程可用，构建可解释的 Broker 心智模型"
date: 2026-02-28T11:03:15+08:00
summary: "用一个可运行的 Go 版本基础消息代理，讲透发布订阅、重试语义、失败契约、吞吐与积压估算，以及从朴素实现到工程可用实现的关键取舍。"
description: "从发布订阅到重试与失败契约，系统讲解基础消息代理的实现与工程边界。"
tags: ["algorithms", "算法", "分布式", "消息队列", "系统设计", "发布订阅", "Broker"]
categories: ["逻辑与算法"]
keywords: ["消息代理", "发布订阅", "消息队列", "Broker", "At-Least-Once", "重试语义"]
readingTime: 18
draft: false
---

## 标题

手写一个基础消息代理：发布、订阅、重试与失败契约

## 副标题 / 摘要

很多“消息队列入门文”只讲概念，不讲失败语义，导致代码能跑但行为不可依赖。本文用一个可运行的最小 Broker，完整讲清发布订阅、重试、幂等与积压控制，并给出从朴素实现到工程可用实现的推导路径。

## 目标读者

- 想从“会用 Kafka/RabbitMQ”走到“理解消息系统核心抽象”的开发者
- 需要设计服务间异步链路、任务分发、事件通知系统的后端工程师
- 正在做系统设计/代码评审，希望把“行为契约”落到实现与测试的人

## 背景 / 动机

同步 RPC 在小系统里简单直接，但在并发上升后会暴露三类瓶颈：

1. **耦合瓶颈**：生产者必须知道消费者地址和可用性
2. **延迟瓶颈**：消费者慢会直接拖慢生产者
3. **可靠性瓶颈**：一次网络抖动就可能丢业务动作

一个最典型的数据点：

- 生产速率 `lambda_in = 300 msg/s`
- 单消费者处理能力 `mu_worker = 80 msg/s`

如果没有缓冲和并行消费，系统会稳定积压，积压增长速度约为：

`backlog_growth = lambda_in - mu_worker = 220 msg/s`

5 分钟后积压约 `220 * 300 = 66,000` 条。  
所以消息代理不是“可有可无的中间层”，而是吞吐和可用性的控制面。

## 快速掌握地图（60-120 秒）

- 问题形状：多生产者、多消费者、异步解耦、可重试交付
- 核心思想一句话：用“主题路由 + 缓冲队列 + 明确失败契约”把调用耦合改成事件耦合
- 什么时候用：跨服务通知、异步任务削峰、批处理触发
- 什么时候避免：强一致事务内联写必须同步确认的场景
- 复杂度头条：单条发布路由 `O(S_t)`（`S_t` 为主题订阅者数），入队 `O(1)`
- 常见失败模式：重试后重复投递，消费者若不幂等会产生重复副作用

## 深化焦点（PDKH）

本文只深化两个概念，不并行扩话题：

1. **概念 A：路由与积压控制**（如何在主题维度做吞吐与背压）
2. **概念 B：失败契约与至少一次投递**（如何让“失败”可被调用方依赖）

PDKH 落地路径（对这两个概念都覆盖）：

- P：重述问题
- D：最小可运行例子
- K：不变量/契约
- H：形式化、复杂度阈值、反例与工程现实

## 主心智模型

把 Broker 想成三个可验证的层：

1. **接入层（Publish API）**：只负责接收消息并放入主题队列
2. **调度层（Dispatcher）**：按主题把消息分发给订阅者
3. **交付层（Delivery）**：执行 handler，处理重试与失败语义

对应不变量：

- `I1`：同一主题队列内，消息出队顺序与入队顺序一致（FIFO）
- `I2`：每个订阅者对同一消息至少收到 1 次交付尝试（At-Least-Once）
- `I3`：超过重试上限后必须进入明确失败路径（日志/死信/告警之一）

## 核心概念与术语

- **Broker**：接收、路由、交付消息的中间层
- **Topic**：逻辑路由键（同一类消息的通道）
- **Subscriber**：注册在某 Topic 上的消费者处理器
- **Attempt**：第几次交付尝试（从 1 开始）
- **At-Least-Once**：至少一次投递，可能重复，不保证恰好一次

关键公式（定义变量）：

- `backlog(t) = produced(t) - acked(t)`
  - `produced(t)`：截止时间 `t` 的累计发布量
  - `acked(t)`：截止时间 `t` 的累计成功处理量
- `workers_required >= ceil(lambda_in / mu_worker)`
  - `lambda_in`：输入速率（msg/s）
  - `mu_worker`：单 worker 稳态处理速率（msg/s）

## 可行性与下界直觉

### 为什么不可能“零成本可靠”

如果你要求：

- 不丢消息
- 不重复消息
- 不阻塞发布端
- 不落盘

这 4 个目标在现实网络和进程故障下不可同时满足。至少要牺牲其中之一（通常牺牲“绝不重复”，选择 At-Least-Once + 幂等）。

### 反例（模型失效场景）

消费者 `handler` 先扣库存再返回超时，Broker 认为失败并重试：

- 第一次：库存已扣减，但返回超时
- 第二次：再次扣减

如果业务没做幂等键（例如 `event_id` 去重），就会发生重复副作用。

## 问题定义（输入/输出/约束）

### 输入

- 发布请求：`Message{ID, Topic, Payload}`
- 订阅注册：`Subscribe(topic, name, handler)`

### 输出

- 对发布者：发布成功/失败（是否入队）
- 对系统：每条消息对每个订阅者的交付结果（成功或超过重试上限失败）

### 约束（本文实现范围）

- 单进程内存版（不含持久化）
- 主题级 FIFO（每个 topic 一条队列）
- 重试上限 `maxRetry` 可配置（默认示例 2）
- 目标：把核心机制讲清，而不是替代生产级 MQ

## 从朴素到可用：基线与瓶颈

### 基线 1：直接函数调用

```text
producer -> consumerA
         -> consumerB
```

- 时间复杂度：`O(S_t)`
- 瓶颈：任何消费者故障直接影响生产者

### 基线 2：发布后立即遍历订阅者执行

- 仍是 `O(S_t)`
- 改进：解耦了地址
- 仍然不足：没有队列缓冲，突发流量无削峰能力

### 关键瓶颈

- 没有“缓冲”就没有“削峰”
- 没有“失败契约”就没有“可依赖行为”

## 关键观察

只要你把“发布成功”定义为“成功入队”（而不是“所有订阅者都执行完成”），系统耦合就会显著下降。之后再通过调度层和交付层分别解决：

- 吞吐问题（积压/背压）
- 失败问题（重试/告警/死信）

## 解释与原理

基础 Broker 能成立，靠的是两个分离：

1. **控制分离**：发布路径只负责“接收并入队”，消费路径负责“执行并确认”  
2. **责任分离**：Broker 负责交付语义，业务 handler 负责幂等与领域副作用

这两个分离让系统从“同步调用的时序耦合”转向“契约驱动的异步耦合”。  
你可以单独优化路由吞吐，而不必同时改业务逻辑；也可以单独升级失败策略，而不改发布 API。

## 实践指南 / 步骤

1. 定义 `Message` 结构，至少包含 `ID`、`Topic`、`Payload`、`Attempt`
2. 建立主题队列 `map[topic]chan Message`，确保 topic 维度缓冲
3. 注册订阅者 `Subscribe(topic, handler)`
4. 发布时只做输入验证 + 入队，不在发布路径执行业务 handler
5. 调度器从 topic 队列取消息并投递给订阅者
6. 交付层封装重试，超过上限走失败路径（日志/死信）
7. 用监控量化 `backlog`、失败率、重试率

## Worked Example（最小追踪）

设定：

- Topic=`order.created`
- 订阅者：`billing`、`inventory`
- `maxRetry = 2`
- 消息：`m-1`

执行轨迹：

1. `publish(m-1)` 成功入队，`backlog=1`
2. 调度器出队 `m-1`，投递给 `billing`（成功）
3. 投递给 `inventory`（第 1 次失败）
4. 重试第 2 次（成功）
5. 该消息对两个订阅者都完成，`backlog` 下降

这个例子说明了两点：

- 发布端不被 `inventory` 的瞬时失败阻塞
- “至少一次”意味着会有重复尝试，消费者必须幂等

## 深化 A：路由与积压控制（PDKH 完整展开）

### P（问题重述）

路由层真正要解决的不是“消息能不能送到”，而是：**当输入速率持续高于消费速率时，如何让系统可预测地退化，而不是随机崩溃**。

### D（最小数值例子）

设单个 topic 的参数为：

- 输入速率 `lambda_in = 120 msg/s`
- 每个 worker 速率 `mu_worker = 35 msg/s`
- worker 数 `W = 2`

则总消费能力 `mu_total = W * mu_worker = 70 msg/s`，积压增长率：

`delta_backlog = lambda_in - mu_total = 50 msg/s`

按秒追踪（假设初始积压为 0）：

| 时间(s) | 累计输入 | 累计消费 | 积压 |
| --- | ---: | ---: | ---: |
| 1 | 120 | 70 | 50 |
| 2 | 240 | 140 | 100 |
| 3 | 360 | 210 | 150 |
| 5 | 600 | 350 | 250 |
| 10 | 1200 | 700 | 500 |

这个表说明：如果不扩容 worker 或降速输入，积压不会“自己恢复”。

### K（不变量/契约）

对路由层给出可测试契约：

- **路由契约-1**：同一 topic 内 FIFO 不破坏
- **路由契约-2**：消息只进入目标 topic 的队列，不跨 topic 污染
- **路由契约-3**：当队列满时必须返回可观测失败（超时/拒绝），不能无限阻塞

### H（形式化 + 阈值）

定义：

- `B_t`：时刻 `t` 的积压
- `I_t`：`t` 到 `t+1` 区间输入数
- `C_t`：`t` 到 `t+1` 区间消费数

状态转移：

`B_{t+1} = max(0, B_t + I_t - C_t)`

扩容阈值可用一个简单规则：

`W_required >= ceil(lambda_p95 / mu_worker_p50)`

其中 `lambda_p95` 用输入速率 95 分位，避免只按平均值规划导致峰值时崩。

### 反例（失败模式）

如果把所有 topic 放进一个全局队列，可能出现“噪声邻居”问题：

- `topic=A` 是高流量低价值日志
- `topic=B` 是低流量高价值支付事件

当 A 突增时，B 会在同一队列后面排队，支付事件延迟异常。  
这是为什么很多系统会做 topic 级别，甚至分区级别隔离。

### 工程现实

路由层优化常见三步：

1. topic 级队列隔离
2. 热点 topic 增加 worker 或分区
3. 入队失败时快速返回并打指标（`publish_timeout_total`）

---

## 深化 B：失败契约与至少一次（PDKH 完整展开）

### P（问题重述）

失败契约要回答的是：**“失败时系统承诺什么”**，而不是“写个 `retry=3` 就完了”。

### D（最小数值例子）

假设消息 `m-9` 处理流程是“扣库存 -> 写订单日志”：

1. attempt=1：库存扣减成功，但日志服务超时
2. Broker 判定失败并重试
3. attempt=2：再次扣减库存

若消费者不幂等，库存被扣两次。这个例子说明：At-Least-Once 必然要求业务层定义幂等契约。

### K（失败契约模板）

建议把每个 handler 的失败语义固化为表：

| 错误类型 | 可重试 | 最大重试 | 失败出口 |
| --- | --- | --- | --- |
| 网络超时 | 是 | 3 | 重试后入告警队列 |
| 参数非法 | 否 | 0 | 直接丢弃并计数 |
| 下游 429 | 是 | 5 | 指数退避 + 限流 |
| 业务冲突（幂等重复） | 否 | 0 | 记成功（幂等命中） |

这张表本质上是契约，而不是实现细节。

### H（形式化 + 阈值）

定义：

- `p_fail`：单次处理失败概率
- `r`：最大重试次数（不含首次）
- `p_drop`：最终失败概率

若假设每次失败独立，近似有：

`p_drop = (p_fail)^(r+1)`

例子：`p_fail = 0.2, r = 2`，则 `p_drop = 0.2^3 = 0.008`。  
但代价是平均尝试次数上升，系统负载增加。

### 反例（错误重试策略）

很多系统把“所有错误都重试 10 次”当默认配置，这会在下游不可用时放大雪崩：

- 下游已过载
- 上游继续重试
- 失败请求数量指数增加

正确做法是“按错误类型分层策略 + 退避 + 熔断”，而不是统一重试。

### 工程现实

至少一次交付要可用，通常还要补三件事：

1. 幂等键（`message_id`/`request_id`）和去重存储
2. 死信队列（DLQ）承接超过重试上限的消息
3. 可追踪链路（日志里至少有 `topic`, `message_id`, `attempt`, `error_code`）

---

## 第二个 Worked Example（失败契约决策追踪）

设某 topic 每分钟 60,000 条消息，观测到：

- 网络超时占失败的 70%
- 参数错误占失败的 20%
- 下游限流占失败的 10%

如果统一重试 3 次，会把参数错误也反复重试，纯浪费资源。  
调整为分层契约后：

- 网络超时：重试 3 次
- 参数错误：不重试，直接失败计数
- 下游限流：重试 5 次且加退避

结果（同流量下）通常是：

- 无效重试显著下降
- 队列积压峰值下降
- 告警噪声下降（错误类型更清晰）

这就是“失败契约先行”的直接工程收益：吞吐不再只靠硬件扩容兜底。

## 正确性（证明草图）

### 不变量

- `I1`：topic 队列内消息顺序不被重排
- `I2`：每个订阅者对每条消息至少执行 1 次 handler
- `I3`：每次失败后 `Attempt` 单调递增，最多 `maxRetry + 1` 次

### 保持性

- 入队只追加到通道尾部，出队按 FIFO，保持 `I1`
- 调度器对每个订阅者都调用 `deliverWithRetry`，保持 `I2`
- `deliverWithRetry` 内部循环自增 attempt 并有上界，保持 `I3`

### 终止性

- 每次交付要么成功返回，要么达到重试上限后失败返回
- 在无无限阻塞 handler 的前提下，单条消息总会结束在“成功”或“失败”状态

## 复杂度分析

记：

- `S_t`：某 topic 订阅者个数
- `R`：最大重试次数（不含首次）

则单条消息的最坏交付成本约为：

- 时间：`O(S_t * (R + 1))`
- 空间（队列）：`O(buffer_per_topic * topic_count)`

若平均失败概率低，平均成本接近 `O(S_t)`。

## 常数因子与工程现实

1. **锁竞争**：高并发 `Subscribe/Publish` 会在路由表锁上竞争
2. **慢消费者拖尾**：串行投递会导致一个慢订阅者拖慢同 topic 整体吞吐
3. **内存风险**：大 `buffer` + 大消息体会放大内存占用
4. **重试风暴**：故障期盲目重试会放大下游压力

一个实用经验阈值：

- 若 `retry_rate > 5%` 且持续 10 分钟，优先限流 + 熔断，而不是继续加重试次数

## 可运行实现（Go）

下面代码是一个单进程最小可运行 Broker，包含：

- 主题队列
- 订阅注册
- 发布入队
- 至少一次投递 + 有上限重试

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

type Message struct {
	ID      string
	Topic   string
	Payload string
	Attempt int
}

type Handler func(context.Context, Message) error

type subscriber struct {
	name    string
	handler Handler
}

type Broker struct {
	mu          sync.RWMutex
	queues      map[string]chan Message
	subscribers map[string][]subscriber
	buffer      int
	maxRetry    int

	closed bool
	stop   chan struct{}
	wg     sync.WaitGroup
}

func NewBroker(buffer, maxRetry int) *Broker {
	if buffer <= 0 {
		buffer = 64
	}
	if maxRetry < 0 {
		maxRetry = 0
	}
	return &Broker{
		queues:      make(map[string]chan Message),
		subscribers: make(map[string][]subscriber),
		buffer:      buffer,
		maxRetry:    maxRetry,
		stop:        make(chan struct{}),
	}
}

func (b *Broker) ensureTopic(topic string) (chan Message, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return nil, errors.New("broker closed")
	}
	if q, ok := b.queues[topic]; ok {
		return q, nil
	}

	q := make(chan Message, b.buffer)
	b.queues[topic] = q

	b.wg.Add(1)
	go b.dispatch(topic, q)

	return q, nil
}

func (b *Broker) Subscribe(topic, name string, h Handler) error {
	if topic == "" || name == "" || h == nil {
		return errors.New("invalid subscribe args")
	}

	_, err := b.ensureTopic(topic)
	if err != nil {
		return err
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	b.subscribers[topic] = append(b.subscribers[topic], subscriber{name: name, handler: h})
	return nil
}

func (b *Broker) Publish(ctx context.Context, msg Message) error {
	if msg.Topic == "" || msg.ID == "" {
		return errors.New("message requires non-empty topic and id")
	}

	q, err := b.ensureTopic(msg.Topic)
	if err != nil {
		return err
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case q <- msg:
		return nil
	}
}

func (b *Broker) dispatch(topic string, q <-chan Message) {
	defer b.wg.Done()

	for {
		select {
		case <-b.stop:
			return
		case msg := <-q:
			b.mu.RLock()
			subs := append([]subscriber(nil), b.subscribers[topic]...)
			b.mu.RUnlock()

			for _, sub := range subs {
				err := deliverWithRetry(context.Background(), sub, msg, b.maxRetry)
				if err != nil {
					log.Printf("topic=%s sub=%s msg=%s dropped after retry: %v", topic, sub.name, msg.ID, err)
				}
			}
		}
	}
}

func deliverWithRetry(ctx context.Context, sub subscriber, msg Message, maxRetry int) error {
	var err error
	for i := 0; i <= maxRetry; i++ {
		attemptMsg := msg
		attemptMsg.Attempt = i + 1
		err = sub.handler(ctx, attemptMsg)
		if err == nil {
			return nil
		}

		if i < maxRetry {
			backoff := time.Duration(50*(i+1)) * time.Millisecond
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}
	}
	return err
}

func (b *Broker) Close() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	close(b.stop)
	b.mu.Unlock()

	b.wg.Wait()
}

func main() {
	broker := NewBroker(16, 2)
	defer broker.Close()

	ctx := context.Background()

	// 模拟一个“首次失败、重试成功”的订阅者
	var mu sync.Mutex
	seen := map[string]int{}

	err := broker.Subscribe("order.created", "billing", func(_ context.Context, m Message) error {
		fmt.Printf("[billing] id=%s attempt=%d payload=%s\n", m.ID, m.Attempt, m.Payload)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	err = broker.Subscribe("order.created", "inventory", func(_ context.Context, m Message) error {
		mu.Lock()
		seen[m.ID]++
		count := seen[m.ID]
		mu.Unlock()

		if m.ID == "m-1" && count == 1 {
			fmt.Printf("[inventory] id=%s attempt=%d simulated fail\n", m.ID, m.Attempt)
			return errors.New("temporary timeout")
		}

		fmt.Printf("[inventory] id=%s attempt=%d ok\n", m.ID, m.Attempt)
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	_ = broker.Publish(ctx, Message{ID: "m-1", Topic: "order.created", Payload: "order=1001"})
	_ = broker.Publish(ctx, Message{ID: "m-2", Topic: "order.created", Payload: "order=1002"})

	// 等待异步处理输出
	time.Sleep(800 * time.Millisecond)
}
```

## 工程应用场景

### 场景 1：订单创建后触发多下游（支付、库存、通知）

- 背景：一个订单事件要触发 3 个系统，不希望强耦合同步调用
- 为什么适配：下游可独立扩缩容，失败可独立重试
- 最小片段：

```go
_ = broker.Publish(ctx, Message{ID: "evt-oid-1", Topic: "order.created", Payload: "oid=1"})
```

### 场景 2：日志异步清洗管道

- 背景：接入层每秒写入数千日志，清洗任务存在抖动
- 为什么适配：队列缓冲可削峰，消费者慢不会直接阻塞接入层
- 最小片段：

```go
for i := 0; i < 1000; i++ {
	_ = broker.Publish(ctx, Message{ID: fmt.Sprintf("log-%d", i), Topic: "log.raw", Payload: "..."})
}
```

### 场景 3：Webhook 事件扇出

- 背景：同一业务事件要推送给多个第三方
- 为什么适配：每个订阅者可独立失败、独立重试
- 最小片段：

```go
_ = broker.Subscribe("user.changed", "webhook-A", handlerA)
_ = broker.Subscribe("user.changed", "webhook-B", handlerB)
```

## 替代方案与取舍

| 方案 | 优点 | 代价 | 适用规模 |
| --- | --- | --- | --- |
| 直接 RPC 链式调用 | 简单、调试直观 | 强耦合、级联失败 | 小规模、低并发 |
| 本文内存 Broker | 轻量、可控、学习成本低 | 无持久化，进程重启丢消息 | 单服务内或学习验证 |
| 生产 MQ（Kafka/RabbitMQ） | 高可靠、可持久化、生态完善 | 运维复杂、协议与语义更重 | 中大规模分布式 |

量化建议：

- 日均消息量 `< 10^5`、单服务内异步解耦：可先用轻量方案
- 日均消息量 `>= 10^7`、跨团队/跨机房链路：应优先生产级 MQ

## 路由参数选型（围绕概念 A）

下面给出一个可以直接抄到评审文档里的选型框架。

### 1）队列缓冲大小怎么估

定义：

- 峰值输入 `lambda_peak`（msg/s）
- 峰值消费 `mu_peak`（msg/s）
- 允许吸收突发时长 `T_burst`（s）

若 `lambda_peak > mu_peak`，建议最小缓冲：

`buffer_min >= (lambda_peak - mu_peak) * T_burst`

示例：

- `lambda_peak = 1200`
- `mu_peak = 900`
- `T_burst = 10`

则 `buffer_min >= 3000`。如果每条消息平均 2 KB，光这部分缓存约 `3000 * 2 KB ≈ 6 MB`（不含元数据）。

### 2）worker 数怎么估

在 topic 内部同质任务下可先用：

`W >= ceil(lambda_p95 / mu_worker_p50)`

为什么是 `p95 / p50` 而不是平均值？  
因为 Broker 最怕峰值拥塞，平均值通常会低估排队风险。

### 3）何时要从“单队列”切到“分区队列”

经验阈值（可作为第一版评审标准）：

- 单 topic 积压长期 > `buffer` 的 60%
- 同一 topic 的处理延迟 P99 > SLA 的 2 倍
- 该 topic 占全系统流量 > 40%

满足任意 2 条，就应评估 topic 分区或热点拆分。

## 失败契约测试矩阵（围绕概念 B）

只写“会重试”不算契约，必须能被测试验证。下面是一组可直接落地的最小测试矩阵：

| 测试编号 | 输入条件 | 预期契约结果 | 验证点 |
| --- | --- | --- | --- |
| T1 | handler 首次失败、次次成功 | 最终成功，`attempt=2` | 日志含同一 `message_id` 两次尝试 |
| T2 | handler 永久失败 | 超过上限后进入失败出口 | 失败计数 + 告警/死信记录 |
| T3 | 参数非法错误 | 不重试，立即失败 | 重试计数不增加 |
| T4 | 发布时队列已满 | `Publish` 返回超时/拒绝 | 发布端可观测错误码 |
| T5 | 重复 `message_id` | 业务幂等，不产生重复副作用 | 幂等存储只写入一次 |

推荐把这 5 个用例作为消息链路改动的回归门槛，避免“功能改完了但语义偷偷变了”。

### 失败契约的最小日志字段

若要排查重试链路，日志至少要有：

- `topic`
- `message_id`
- `subscriber`
- `attempt`
- `error_code`
- `retryable`
- `final_state`（success / dropped）

缺任何一个字段，线上定位成本都会显著上升。

### 交付语义边界（不要混淆）

实现消息系统时最常见的沟通误差，是把三种语义混为一谈：

- **At-Most-Once（至多一次）**：可能丢，不重试，不重复  
- **At-Least-Once（至少一次）**：不轻易丢，会重试，可能重复  
- **Exactly-Once（恰好一次）**：理论目标很强，通常需要跨组件事务与幂等协同，成本最高

本文实现明确属于 **At-Least-Once**。  
所以“重复投递”不是 bug，而是契约的一部分；真正的 bug 是业务层没有声明并实现幂等契约。

在工程评审里，建议每条链路都显式写一行：

`delivery_semantics = at_most_once | at_least_once | exactly_once(target)`  

不写这行，后续排障时几乎一定会出现“到底算系统错还是业务错”的责任争议。

### 一个常见反模式

反模式：在 handler 里 `panic` 后由框架兜底吞掉，Broker 端认为“处理成功”。  
后果：你失去了失败可观测性，契约被悄悄破坏。

修正策略：

1. handler 内部统一把异常转换成显式错误返回
2. Broker 只以“返回值”判断成功/失败
3. 所有失败路径都进入同一指标和日志出口

## 常见问题与注意事项

1. **为什么会重复消费？**  
   At-Least-Once 天生允许重试重复，必须依赖业务幂等键去重。

2. **发布成功是不是代表业务成功？**  
   不是。发布成功通常只代表“成功入队”。业务成功取决于消费者处理结果。

3. **如何避免消息无限重试？**  
   设重试上限 + 失败出口（死信/告警/人工补偿）。

4. **FIFO 是否跨订阅者全局成立？**  
   本文实现只保证 topic 队列出队顺序；跨订阅者并行执行时不保证全局完成顺序。

5. **反压怎么做？**  
   队列满时 `Publish` 应返回超时或拒绝，而不是无限阻塞。

## 最佳实践与建议

- 先写失败契约：哪些错误重试、重试几次、失败落哪里
- 让每条消息带业务幂等键（`event_id` / `request_id`）
- 监控三件事：积压长度、重试率、最终失败率
- 把“发布成功”与“业务完成”分成两类状态，不混用
- 对慢消费者做隔离（独立队列或并行 worker）
- 在重试间隔上加退避，避免故障放大

### 压测与验收清单（上线前建议最少做一次）

下面这组数据建议在预发环境至少跑 30 分钟：

1. **稳态吞吐测试**  
   以目标流量 `1.2x` 持续压测，观察 `backlog` 是否收敛到稳定区间。  
   验收线示例：`backlog_p95 < buffer * 0.5`。

2. **故障注入测试（消费者超时）**  
   人工让一个订阅者超时 5 分钟，观察：\n   - 发布端是否仍可返回可观测错误（而非卡死）\n   - 重试率是否在预期范围内（例如 < 15%）\n   - 故障恢复后积压是否在 10 分钟内回落

3. **重复投递幂等测试**  
   人工重复投递同一 `message_id` 100 次，确认业务副作用只发生 1 次。  
   若不是 1 次，说明“至少一次”语义与业务层幂等契约没有闭合。

4. **失败出口测试（超上限）**  
   构造永久失败消息，确认超过重试上限后进入统一失败出口（死信或告警），并带完整字段：`topic/message_id/subscriber/attempt/error_code`。

通过这 4 组测试，你才能证明不是“实现看起来对”，而是“契约在压力和故障下仍然成立”。

## 小结 / 结论

核心收获（可直接落地）：

1. Broker 的本质是把“同步依赖”改成“异步契约”。
2. 发布/订阅模型要先定义失败语义，再谈代码实现。
3. At-Least-Once 不是缺点，但它强制你实现幂等。
4. 吞吐问题可以先用 `lambda_in` 与 `mu_worker` 粗估是否会积压。
5. 从最小实现出发，先把不变量和契约跑通，再升级到持久化系统。

## 参考与延伸阅读

- Martin Kleppmann, *Designing Data-Intensive Applications*
- Kafka Documentation: Producers / Consumers / Delivery Semantics
- RabbitMQ Tutorials: Publish/Subscribe, Work Queues

## 元信息

- **阅读时长**：约 18 分钟
- **标签**：消息代理、发布订阅、重试语义、分布式
- **SEO 关键词**：消息代理, 发布订阅, Broker, At-Least-Once, 消息队列
- **元描述**：通过可运行 Go 示例讲解基础消息代理的发布订阅、重试语义、失败契约与工程权衡。

## 行动号召（CTA）

下一步可以直接做两件事：

1. 给当前实现加一个内存死信队列（DLQ），把超过重试上限的消息单独存放并可查询。
2. 给消费者增加幂等存储（按 `message_id` 去重），验证重复投递不会产生重复副作用。
