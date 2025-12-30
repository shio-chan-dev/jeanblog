---
title: "How to Use wrk for Load Testing"
date: 2025-10-22
draft: false
---

# Load Testing APIs with wrk (Detailed Guide)

> This article explains how to use `wrk` on Ubuntu to stress-test backend APIs (Flask, FastAPI, Spring Boot, etc.) and interpret the results.

---

## 1. What is wrk?

[`wrk`](https://github.com/wg/wrk) is a modern, high-performance HTTP benchmarking tool written in C. Key features:

- **High concurrency**: thousands of concurrent connections
- **Multi-threaded**: uses multiple CPU cores
- **Lua scripting**: for custom headers, bodies, tokens
- **Faster than Apache Benchmark (ab)**: lighter and more stable

---

## 2. Install wrk

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install wrk -y
```

Verify:

```bash
wrk --version
```

Expected output:

```
wrk 4.2.0 [epoll]
```

---

## 3. Quick start

Suppose your service is at:

```
http://192.168.1.224:5000/api/tenders
```

Run:

```bash
wrk -t4 -c100 -d30s http://192.168.1.224:5000/api/tenders
```

### Parameters

| Flag | Meaning |
| --- | --- |
| `-t4` | 4 threads (use multi-core CPU) |
| `-c100` | 100 concurrent connections |
| `-d30s` | 30 seconds duration |
| last arg | target URL |

---

## 4. Sample output explained

Example output:

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

### Metrics

| Metric | Meaning | Example | Notes |
| --- | --- | --- | --- |
| **Latency** | Average response time | 1.12s | Slow if > 1s |
| **Req/Sec** | Requests per thread per second | 22.88 | Depends on thread count |
| **Requests/sec** | Total QPS | 81.49 | Throughput |
| **Transfer/sec** | Data per second | 0.90MB | Bandwidth usage |
| **Timeouts** | Timed-out requests | 2 | Indicates delays |

> In general:
> - Excellent: latency < 200ms
> - OK: 200-800ms
> - Slow: > 1s

---

## 5. Tips to improve concurrency

### 1) Use a production server (Flask example)

Do not use `app.run()`. Use **Gunicorn**:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

- `-w 4`: 4 worker processes (recommended `2 * CPU + 1`)
- Improves concurrency and stability

---

### 2) Increase async throughput (I/O-bound APIs)

```bash
gunicorn -w 4 -k gevent -b 0.0.0.0:5000 run:app
```

`-k gevent` uses async workers to handle many waiting requests.

---

### 3) Reduce response size

Large responses consume network bandwidth. Suggestions:

- Return only required fields
- Enable gzip (Nginx or Flask plugins)

---

## 6. Advanced: Lua scripts for custom requests

Lua scripts can do:

- Custom headers and tokens
- POST JSON bodies
- Randomized parameters

Example `post.lua`:

```lua
wrk.method = "POST"
wrk.body   = '{"keyword":"test"}'
wrk.headers["Content-Type"] = "application/json"
```

Run:

```bash
wrk -t4 -c100 -d30s -s post.lua http://127.0.0.1:5000/api/search
```
