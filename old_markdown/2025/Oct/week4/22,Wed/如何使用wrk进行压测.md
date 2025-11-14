
# 🚀 使用 wrk 对接口进行高性能压力测试（超详细教程）

> 本文介绍如何在 Ubuntu 环境中使用 `wrk` 对后端接口（如 Flask / FastAPI / Spring Boot 等）进行高并发压力测试，并结合结果分析性能瓶颈。

---

## 🧰 一、什么是 wrk？

[`wrk`](https://github.com/wg/wrk) 是一个现代化、高性能的 HTTP 压测工具，由 C 语言编写，具有以下特点：

* **高并发能力强**：支持成千上万的并发连接
* **支持多线程**：充分利用多核 CPU
* **可自定义 Lua 脚本**：适合复杂场景（如自定义请求头、Body、Token 等）
* **比 Apache Benchmark (ab)** 更轻量、更快、更稳定

---

## ⚙️ 二、安装 wrk

在 Ubuntu / Debian 上安装：

```bash
sudo apt update
sudo apt install wrk -y
```

验证安装是否成功：

```bash
wrk --version
```

输出类似：

```
wrk 4.2.0 [epoll]
```

表示安装成功 ✅

---

## 🧪 三、快速开始压测

假设你的服务运行在：

```
http://192.168.1.224:5000/api/tenders
```

运行：

```bash
wrk -t4 -c100 -d30s http://192.168.1.224:5000/api/tenders
```

### 参数说明：

| 参数      | 含义                 |
| ------- | ------------------ |
| `-t4`   | 启动 4 个线程（利用多核 CPU） |
| `-c100` | 模拟 100 个并发连接       |
| `-d30s` | 持续压测 30 秒          |
| 最后一个参数  | 目标 URL             |

---

## 📊 四、示例输出结果解读

假设输出如下：

```
Running 30s test @ http://192.168.1.224:5000/api/tenders
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     1.12s   248.83ms   1.99s    85.59%
    Req/Sec    22.88     14.29    90.00     77.73%
  2452 requests in 30.09s, 27.02MB read
  Socket errors: connect 0, read 0, write 0, timeout 2
Requests/sec:     81.49
Transfer/sec:      0.90MB
```

### 结果分析：

| 指标               | 含义              | 示例值    | 说明        |
| ---------------- | --------------- | ------ | --------- |
| **Latency**      | 每个请求平均响应时间      | 1.12s  | 响应较慢（>1s） |
| **Req/Sec**      | 每个线程每秒请求数       | 22.88  | 与线程数有关    |
| **Requests/sec** | 整体 QPS（每秒处理请求数） | 81.49  | 表示服务吞吐量   |
| **Transfer/sec** | 每秒传输数据量         | 0.90MB | 网络带宽占用情况  |
| **Timeouts**     | 超时请求数           | 2      | 稍有请求延迟过长  |

> 🔍 一般情况下：
>
> * 优秀接口：延迟 < 200ms
> * 中等接口：200–800ms
> * 过慢接口：>1s

---

## ⚡ 五、提高并发性能的实用技巧

### ✅ 1. 使用生产级服务器（Flask 示例）

不要用 Flask 的 `app.run()`。
改用 **Gunicorn** 启动：

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

* `-w 4`：4 个 worker 进程（推荐：`2 * CPU核数 + 1`）
* 能显著提升并发能力与稳定性

---

### ✅ 2. 增加异步处理能力（适合 I/O 密集型接口）

```bash
gunicorn -w 4 -k gevent -b 0.0.0.0:5000 run:app
```

`-k gevent` 使用异步 worker 模型，可同时处理大量等待中的请求。

---

### ✅ 3. 减少响应体大小

压测时，每个请求的响应体越大，网络吞吐越受限。
建议：

* 只返回必要字段
* 启用 Gzip 压缩（Nginx 或 Flask 插件）

---

## 📈 六、高级用法：Lua 脚本自定义请求

你可以用 Lua 脚本实现：

* 自定义请求头 / Token
* POST JSON 请求
* 参数随机化

示例 `post.lua`：

```lua
wrk.method = "POST"
wrk.body   = '{"keyword":"test"}'
wrk.headers["Content-Type"] = "application/json"
```

运行：

```bash
wrk -t4 -c100 -d30s -s post.lua http://127.0.0.1:5000/api/search
```

