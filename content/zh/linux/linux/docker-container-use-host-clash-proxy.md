---
title: "Docker 容器如何使用宿主机 Clash 代理"
date: 2026-05-15T23:50:00+08:00
draft: false
description: "记录一次 Docker 容器无法直连 OpenAI OAuth 的排障过程，并说明如何通过 host.docker.internal 复用宿主机 Clash 代理。"
tags: ["Docker", "Clash", "代理", "OpenAI OAuth", "网络排障"]
categories: ["Linux"]
keywords: ["Docker 容器访问宿主机代理", "host.docker.internal", "Clash allow-lan", "OpenAI OAuth 代理"]
---

## 副标题 / 摘要

宿主机能访问 OpenAI，不代表 Docker 容器也能访问。本文记录一次 sub2api 容器无法完成 OpenAI OAuth 的排障过程，并给出一种不绑定 Docker 网段 IP 的配置方式。

## 目标读者

- 使用 Docker Compose 部署服务的开发者
- 在服务器上使用 Clash、Mihomo 或类似代理的用户
- 遇到容器内访问 OpenAI、GitHub、Google 等服务超时的人
- 需要理解 `127.0.0.1`、Docker 网桥和宿主机代理关系的人

## 背景 / 动机

问题表面上是 sub2api 后台提示：

```text
未设置代理，当前服务器无法直连 OpenAI，导致 OpenAI OAuth 请求失败。
请先选择可访问 OpenAI 的代理后重试；如果授权码已失效，请重新生成授权链接。
```

容易产生的疑问是：我的电脑明明可以访问 OpenAI，Codex 也能用，为什么 Docker 里的服务不行？

原因是 Codex、浏览器、终端和 Docker 容器不一定走同一条网络路径。宿主机上的程序可以使用 `127.0.0.1:7890` 代理，但容器里的 `127.0.0.1` 指向的是容器自己，不是宿主机。

## 核心概念

### 宿主机的 127.0.0.1 不是容器的 127.0.0.1

如果 Clash 只监听：

```text
127.0.0.1:7890
```

那么只有宿主机本机进程能访问它。Docker 容器访问 `127.0.0.1:7890` 时，访问的是容器内部的 7890 端口，通常什么都没有。

### Docker Compose 服务名只在 Compose 网络内有效

例如 `postgres`、`redis` 这类服务名依赖 Docker 内置 DNS。它解决的是容器之间互相访问，不解决容器访问宿主机代理的问题。

### `host.docker.internal` 是更稳定的宿主机别名

直接写 Docker 网关 IP，例如 `172.24.0.1`，能用但不够稳。Compose 项目、网络重建后，网段可能变化。

更好的方式是在服务里加：

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

这样容器内可以用 `host.docker.internal` 访问宿主机网关。

## 实践指南 / 步骤

### 1. 让 Clash 允许 Docker 容器访问

Clash 配置里需要允许局域网或 Docker 网桥访问代理端口：

```yaml
port: 7890
socks-port: 7891
allow-lan: true
bind-address: '*'
```

重启 Clash：

```bash
sudo systemctl restart clash
```

检查监听状态：

```bash
ss -ltnp | rg ':(7890|7891)\b'
```

成功时应该看到类似：

```text
LISTEN 0 4096 *:7890 *:*
LISTEN 0 4096 *:7891 *:*
```

如果仍然是：

```text
127.0.0.1:7890
127.0.0.1:7891
```

说明容器仍然访问不到。

### 2. 在 Docker Compose 中加入宿主机别名

在需要访问宿主机代理的服务下加入：

```yaml
services:
  sub2api:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

修改后验证 Compose 配置：

```bash
docker compose config | rg -n "extra_hosts|host.docker.internal" -C 2
```

让正在运行的容器重新创建，使 `extra_hosts` 生效：

```bash
docker compose up -d --force-recreate --no-deps sub2api
```

### 3. 在容器内验证代理是否可达

先验证容器能解析宿主机别名：

```bash
docker exec sub2api getent hosts host.docker.internal
```

再验证代理端口是否能连接：

```bash
docker exec sub2api nc -vz -w 3 host.docker.internal 7890
```

最后通过代理访问 OpenAI：

```bash
docker exec sub2api curl -I \
  --connect-timeout 5 \
  --max-time 12 \
  -x http://host.docker.internal:7890 \
  https://auth.openai.com/oauth/token
```

如果看到：

```text
HTTP/1.1 200 Connection established
```

说明容器已经能通过宿主机代理发起 HTTPS 连接。后面的 `405`、`421` 等 HTTP 状态不一定是错误，因为这里用的是 `HEAD` 请求，只是为了验证网络链路。

### 4. 在应用后台绑定代理

以 sub2api 为例，后台代理配置可以填：

```text
协议: http
主机: host.docker.internal
端口: 7890
```

创建或编辑 OpenAI OAuth 账号时，需要选择这个代理。只配置 Docker 网络别名还不够，因为 OpenAI OAuth 这类账号请求通常会使用账号绑定的代理。

如果之前授权码已经失败过，建议重新生成授权链接。OAuth 授权码通常有时效，失败后继续复用可能会遇到授权码过期。

## 可运行示例

完整的 Compose 片段如下：

```yaml
services:
  sub2api:
    image: weishaw/sub2api:latest
    container_name: sub2api
    restart: unless-stopped
    ports:
      - "0.0.0.0:8080:8080"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Clash 侧关键配置：

```yaml
port: 7890
socks-port: 7891
allow-lan: true
bind-address: '*'
```

容器内验证命令：

```bash
docker exec sub2api curl -I \
  -x http://host.docker.internal:7890 \
  https://api.openai.com
```

## 解释与原理

这类问题的核心不是 OpenAI OAuth 本身，而是网络边界。

宿主机程序访问代理时，路径通常是：

```text
宿主机进程 -> 127.0.0.1:7890 -> Clash -> 外网
```

Docker 容器访问代理时，不能直接复用宿主机的 `127.0.0.1`。正确路径应该是：

```text
容器进程 -> host.docker.internal:7890 -> 宿主机 Clash -> 外网
```

`extra_hosts` 的作用是给容器注入一个稳定名称，避免直接写当前 Docker 网关 IP。Clash 的 `allow-lan: true` 和 `bind-address: '*'` 则让这个代理端口不再只绑定宿主机回环地址。

## 常见问题与注意事项

### 1. 为什么 Codex 能用，Docker 服务不能用？

Codex 运行在宿主机环境里，可能继承了：

```text
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
all_proxy=socks5://127.0.0.1:7890
```

Docker 容器不会自动继承这些代理设置。即使继承了，容器里的 `127.0.0.1` 也不是宿主机。

### 2. 可以用 `network_mode: host` 吗？

可以，但通常不建议作为首选。Host 网络会改变容器网络隔离模型，也可能影响 Compose 服务名解析、端口映射和服务之间的访问方式。

对于只需要访问宿主机代理的场景，`host.docker.internal` 更小、更清晰。

### 3. 只设置 `HTTP_PROXY` 和 `HTTPS_PROXY` 行不行？

要看应用是否使用环境变量代理。很多应用的上游请求会走账号级代理、数据库里的代理配置，或者自定义 HTTP 客户端。

在 sub2api 这类场景中，OpenAI OAuth 账号最好明确绑定后台代理记录。

### 4. 开启 `allow-lan` 有没有风险？

有。`*:7890` 可能被局域网甚至公网访问，取决于服务器防火墙和网络环境。

至少应该限制防火墙，只允许本机和 Docker 网段访问代理端口。不要把 Clash 的代理端口直接暴露给公网。

## 最佳实践与建议

- 不要在容器里填 `127.0.0.1:7890`，除非容器内真的运行了代理。
- 不要写死 Docker 网关 IP，优先使用 `host.docker.internal`。
- Clash 只需要开放给可信网络，不要无保护暴露到公网。
- 应用后台有账号代理配置时，要把账号绑定到代理上。
- OAuth 授权码失败后，重新生成授权链接再试。

## 小结 / 结论

宿主机能访问 OpenAI，不代表 Docker 容器也能访问。排查时要分清三层：

1. 宿主机代理是否监听到 Docker 可访问的地址。
2. Docker 容器是否能解析并访问宿主机代理。
3. 应用账号是否实际绑定了这个代理。

把这三层打通后，容器内服务就可以稳定复用宿主机 Clash 代理。

## 参考与延伸阅读

- Docker Compose `extra_hosts`
- Docker `host-gateway`
- Clash / Mihomo `allow-lan` 与 `bind-address`
- OAuth 授权码流程

## 元信息

- **阅读时长**：8~10 分钟
- **标签**：Docker、Clash、代理、OpenAI OAuth
- **SEO 关键词**：Docker 容器访问宿主机代理、host.docker.internal、Clash allow-lan
- **元描述**：记录 Docker 容器无法直连 OpenAI OAuth 的排障过程，以及如何通过 host.docker.internal 使用宿主机 Clash 代理。

## 行动号召（CTA）

下次遇到“宿主机能访问、容器不能访问”的问题，先在容器内跑一次 `curl -x` 和 `getent hosts`，把网络路径验证清楚。
