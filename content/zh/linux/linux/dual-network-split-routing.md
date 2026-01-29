---
title: "一台电脑同时连公司内网和手机外网：双网络分流实战"
date: 2026-01-29T16:00:40+08:00
draft: false
categories: ["linux", "network"]
tags: ["双网卡", "路由分流", "热点", "VPN", "WireGuard", "Tailscale"]
description: "讲清楚一台电脑如何同时访问公司内网与手机外网，覆盖 USB 共享、双 WiFi、静态路由与 VPN 分流。"
keywords: ["双网络", "路由分流", "热点上网", "WireGuard", "Split Tunnel"]
---

### **标题**

一台电脑同时连公司内网和手机外网：双网络分流实战

---

### **副标题 / 摘要**

想用公司内网访问内部服务，又希望互联网走手机热点？
本文给出 3 套可落地方案：USB 共享、双网卡分流、VPN Split Tunnel，
并提供 Windows/macOS/Linux 的实操步骤。

---

### **目标读者**

* 需要访问公司内网，但不想走公司外网出口的开发者/运维
* 想在一个电脑上同时使用两条网络链路的技术人员
* 需要稳定分流、减少网络切换成本的同学

---

### **背景 / 动机**

你希望达到的效果是：

* **公司内网服务**（Gitlab、内网 API、数据库）走公司网络
* **外网互联网**（搜索、下载、第三方服务）走手机热点

但普通电脑默认只能有**一个默认网关**，
所以要么全部走公司网，要么全部走手机网。
这也是需要“分流”的原因。

---

### **核心概念**

* **网卡（NIC）**：每条网络连接对应一个网卡（WiFi、USB、网线）
* **默认路由**：系统不知道去哪就走默认网关
* **最长前缀匹配**：更具体的网段路由会优先生效
* **分流（Split Routing / Split Tunnel）**：内网走公司，外网走手机

---

## 实践指南 / 步骤

下面按稳定性从高到低给出三种方案：

### 方案 A：公司内网 WiFi + 手机 USB 共享（最稳定）

**适用场景**：只有一张 WiFi 网卡，想要最省事的分流方式。

1. 连接公司 WiFi（内网）
2. 手机开启 USB 共享网络，连接电脑
3. 系统通常会把 USB 网络作为默认外网

优点：稳定、无需额外硬件；
缺点：需要 USB 连接手机。

---

### 方案 B：双 WiFi（USB 无线网卡）

**适用场景**：必须同时连接两个 WiFi。  

1. 电脑内置 WiFi 连接公司内网
2. 购买一个 USB WiFi 作为第二网卡
3. 第二网卡连接手机热点

优点：无线方便；
缺点：需要额外硬件，易受干扰。

---

### 方案 C：VPN Split Tunnel（最干净）

**适用场景**：只用手机热点上网，但仍需访问公司内网。  

1. 电脑只连接手机热点
2. 用 WireGuard/Tailscale 连接公司内网
3. 设置 AllowedIPs 只包含内网网段

优点：最清晰的分流方式；
缺点：需要公司内网有 VPN 入口或引入 Tailscale。

#### C-1）无公网 IP 场景：Tailscale 子网路由（最现实）

如果公司服务器在 NAT 后面、没有公网入口，**Tailscale 是最省事的方案**。

公司服务器安装并登录：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --advertise-routes=192.168.1.0/24
```

然后在 Tailscale 控制台 **Routes** 页面批准该网段（Approve/Enable）。
你的电脑登录同一账号后，就能在手机热点下访问内网：

```
192.168.1.x
```

**登录成功后下一步怎么做（关键清单）**

1) 确认设备在线：

```bash
tailscale status
```

2) 开启转发（必须）：

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo sysctl -w net.ipv6.conf.all.forwarding=1
```

3) 如内网仍不通，补 NAT（按需）：

```bash
sudo iptables -t nat -A POSTROUTING -o <内网网卡> -j MASQUERADE
```

> `<内网网卡>` 一般是 `eno2`/`eth0`/`ens*`，可用 `ip a` 查看。

#### C-2）有公网入口：WireGuard 端口映射

如果你能控制公司网关/路由器，可以做端口映射：

```
公网 UDP 51820 -> 192.168.1.224:51820
```

这样你的电脑就可以用标准 WireGuard 连接内网。

#### C-3）自建穿透：FRP / 反向隧道

如果没有公网 IP、又无法用 Tailscale，可以用 FRP：

* 公司服务器主动连公网 VPS
* 外网通过 VPS 再进公司内网

这类方案更“运维化”，但可控性强，适合自建。

---

## 可运行示例（静态路由分流）

下面是“只把内网走公司网络”的静态路由示例。

### Windows

```bat
:: 查网关：ipconfig
:: 假设公司内网网关是 192.168.1.1，内网段是 192.168.0.0/16
route add 192.168.0.0 mask 255.255.0.0 192.168.1.1 metric 5 -p
```

### macOS

```bash
# 假设公司内网网关是 192.168.1.1
sudo route -n add -net 192.168.0.0/16 192.168.1.1
```

### Linux

```bash
# 假设公司内网网关是 192.168.1.1，网卡是 wlan0
sudo ip route add 192.168.0.0/16 via 192.168.1.1 dev wlan0
```

---

## 解释与原理（为什么这么做）

路由选择遵循“最长前缀匹配”原则：

* **192.168.0.0/16** 比 **0.0.0.0/0** 更具体
* 所以内网地址会优先走公司网关
* 其他流量仍走默认路由（手机热点）

这就是分流的本质：
**给内网加“更具体的路由”，默认外网不动。**

---

## E — Engineering（工程应用）

### 场景 1：开发环境访问内网服务

**背景**：需要访问公司 Gitlab / 内部 API。  
**为什么适用**：内网走公司网，避免外网出口受限。  

```bash
curl http://192.168.1.10/api/health
```

### 场景 2：手机热点上外网

**背景**：公司外网受限或速度慢。  
**为什么适用**：默认路由走手机热点，外网稳定可控。  

```bash
curl ifconfig.me
```

### 场景 3：VPN Split Tunnel

**背景**：只连手机热点，但需要内网资源。  
**为什么适用**：VPN 只接管内网段，不影响外网。  

```ini
# WireGuard Client (示例)
[Interface]
Address = 10.200.200.2/24
PrivateKey = <client_private_key>
DNS = 192.168.1.1

[Peer]
PublicKey = <server_public_key>
Endpoint = <public_ip_or_tailscale_ip>:51820
AllowedIPs = 192.168.0.0/16
PersistentKeepalive = 25
```

---

## 常见问题与注意事项

1. **只有一张 WiFi 卡能连两个 WiFi 吗？**  
   通常不行，需要额外 USB 无线网卡。

2. **加了路由但内网还是不通？**  
   检查公司网关是否正确，确认内网服务是否允许访问。

3. **内网域名解析失败怎么办？**  
   需要使用公司内网 DNS，或在 VPN 配置里指定 DNS。

4. **路由重启后失效？**  
   Windows 用 `-p` 添加永久路由，Linux 需写入网络配置。

5. **公司是否允许自建 VPN？**  
   先确认公司安全政策，避免违规。

---

## 最佳实践与建议

* 优先选择 **WiFi + USB 共享**，最稳定且最少配置。
* 内网访问建议显式添加路由，避免走错出口。
* VPN 使用 **Split Tunnel**，避免外网流量走公司。
* 记录当前路由表，便于排查（Windows: `route print`，Linux: `ip route`）。

---

## 小结 / 结论

一台电脑同时连公司内网和手机外网的核心是：
**两条链路 + 明确路由分流**。
最简单稳定的方案是“公司 WiFi + 手机 USB 共享”，
更工程化的方案是“VPN Split Tunnel”。

---

## 参考与延伸阅读

- https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/route_ws2008
- https://man7.org/linux/man-pages/man8/ip-route.8.html
- https://man.openbsd.org/route
- https://www.wireguard.com/
- https://tailscale.com/kb/

---

## 元信息

- **阅读时长**：约 10 分钟
- **标签**：双网卡、路由分流、VPN
- **SEO 关键词**：双网络, 路由分流, Split Tunnel, WireGuard, Tailscale
- **元描述**：一台电脑同时访问公司内网与手机外网的实战指南，含多平台路由分流方案。

---

## 行动号召（CTA）

如果你愿意提供你的系统（Windows / macOS / Linux）和内网网段，我可以直接给你可用的分流脚本。
