---
title: "WireGuard Split Tunnel 实战：手机热点上外网，同时访问公司内网"
date: 2026-01-29T16:15:21+08:00
draft: false
categories: ["linux", "network"]
tags: ["wireguard", "vpn", "split-tunnel", "内网访问", "端口映射", "NAT", "Windows"]
description: "从 0 搭建 WireGuard 内网 VPN，并用 Split Tunnel 实现“外网走手机热点、内网走 VPN”；覆盖公网/无公网两种部署与排错清单。"
keywords: ["WireGuard", "Split Tunnel", "内网访问", "端口映射", "NAT", "手机热点", "VPN"]
---

### **标题**

WireGuard Split Tunnel 实战：手机热点上外网，同时访问公司内网

---

### **副标题 / 摘要**

你想在外网（手机热点）正常上网，同时访问公司内网（192.168.x.x）服务。
最干净的方式是：**外网默认走手机热点，只有内网网段走 WireGuard（Split Tunnel）**。

---

### **目标读者**

* 需要在外网访问公司内网服务（Gitlab/内部 API/数据库）的开发者
* 公司内网没有官方 VPN，或现有 VPN 体验差
* 希望“分流”：外网不走公司出口，内网安全可控

---

### **背景 / 动机**

很多公司内网服务只在私网开放（如 `192.168.1.0/24`）。
当你在外面用手机热点上网时：

* 外网访问没问题
* 但内网地址不可达

如果你把电脑同时连两条网络（公司 WiFi + 手机热点），“能用但不干净”：
路由/DNS 冲突、稳定性差、切换成本高。

WireGuard 的价值在于：

* 只打通你需要的内网网段（Split Tunnel）
* 外网仍走你自己的手机出口
* 连接快、配置简单、性能好

---

### **核心概念**

* **WireGuard**：现代 VPN 协议与实现，配置简洁、性能优秀。
* **Peer**：对端节点（客户端/服务器）。
* **AllowedIPs（关键）**：决定哪些流量走 VPN。
* **Split Tunnel（分流）**：AllowedIPs 只写内网网段，不接管默认路由。
* **NAT / 端口映射**：服务器在内网（192.168.*）时，外网直连需要网关转发 UDP 端口。

---

## 思维推导（从“想要双网”到“正确分流”）

1) 需求：公司内网访问 + 手机热点上外网。
2) 朴素解：同时连两张网卡，靠系统自动选路由 → 不稳定。
3) 关键观察：你的目标不是“同时连两张网”，而是“**只让内网走公司通道**”。
4) 方法选择：WireGuard Split Tunnel：仅路由 `192.168.x.x` 走 VPN。
5) 约束：WireGuard 需要一个**外网可达的 Endpoint**；如果公司服务器在 NAT 后面，需要端口映射或中转方案。

---

## A — Algorithm（题目与算法）

### 题目还原

> 电脑连手机热点上网，但还能访问公司内网（如 192.168.1.0/24）。

### 解法要点

* 让客户端的默认路由仍然是手机热点
* 为公司内网网段添加一条“更具体”的路由，走 WireGuard

在 WireGuard 里，**这条路由由 `AllowedIPs` 决定**。

---

## C — Concepts（核心思想）

### WireGuard 的最小心智模型

* 你有两类 IP：
  * **公司内网 IP 段**：例如 `192.168.1.0/24`
  * **WireGuard 虚拟网段**：例如 `10.200.200.0/24`

* 客户端通过 UDP 连接服务器 `Endpoint`，协商后建立加密隧道。
* 当你的流量命中 `AllowedIPs` 指定网段时，流量会进入隧道。

### Split Tunnel 的关键

* ❌ 全流量 VPN（不适合你的目标）：

```ini
AllowedIPs = 0.0.0.0/0
```

* ✅ 只接管公司内网（你要的）：

```ini
AllowedIPs = 192.168.1.0/24
```

---

## 实践指南 / 步骤（从 0 部署）

下面以公司内网 `192.168.1.0/24` 为例，你可替换成自己的网段。

### Step 0：确认公司服务器是否“外网可达”

在公司服务器上执行：

```bash
ip a
curl ifconfig.me
```

* `ip a` 看到 `192.168.*`：说明服务器在内网
* `curl ifconfig.me` 看到公网 IP：通常是公司出口 NAT（不代表这台服务器可被外网直连）

若服务器无公网入口，你需要：

* 端口映射（网关把 UDP 51820 转发到该服务器），或
* 用公网 VPS 做中转（后文提供方案），或
* 改用 Tailscale（更省事，但本文主讲 WireGuard）

---

### Step 1：在公司内网服务器安装 WireGuard（Linux）

Ubuntu/Debian：

```bash
sudo apt update
sudo apt install -y wireguard
```

---

### Step 2：生成密钥（Server & Client）

在服务器上生成：

```bash
wg genkey | tee server.key | wg pubkey > server.pub
wg genkey | tee client.key | wg pubkey > client.pub
```

---

### Step 3：配置服务器 `/etc/wireguard/wg0.conf`

> 说明：下方 `<...>` 用你自己的值替换；不要把私钥提交到仓库。

```ini
[Interface]
Address = 10.200.200.1/24
ListenPort = 51820
PrivateKey = <server_private_key>

# 开启转发 + NAT（让客户端能访问 192.168.1.0/24）
PostUp = sysctl -w net.ipv4.ip_forward=1
PostUp = iptables -t nat -A POSTROUTING -o eno2 -j MASQUERADE
PostDown = iptables -t nat -D POSTROUTING -o eno2 -j MASQUERADE

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.200.200.2/32
```

注意：

* `eno2` 是你公司内网网卡名（用 `ip a` 查看后替换）
* 如果你希望多台客户端接入，每台一个 `[Peer]` 并分配不同 `10.200.200.x`

---

### Step 4：开启转发（永久生效）

```bash
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

---

### Step 5：启动 WireGuard

```bash
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0
sudo wg
```

---

### Step 6：让外网能连到公司服务器（端口映射）

如果公司服务器只有 `192.168.1.224` 这类内网地址，外网无法直接连到它。
你必须在公司网关/路由器上配置端口映射：

```
公网 UDP 51820 -> 192.168.1.224:51820
```

> 这是很多公司做不到的关键点：你需要能控制网关，或请网络管理员协助。

---

### Step 7：客户端（Windows/macOS/Linux）配置

创建客户端配置 `wg-client.conf`：

```ini
[Interface]
Address = 10.200.200.2/24
PrivateKey = <client_private_key>

[Peer]
PublicKey = <server_public_key>
Endpoint = <公司公网IP或域名>:51820

# ✅ Split Tunnel：只把公司内网走 VPN
AllowedIPs = 192.168.1.0/24
PersistentKeepalive = 25
```

说明：

* `PersistentKeepalive=25` 对 NAT 环境非常重要（保持映射不超时）
* 只写 `192.168.1.0/24`，外网仍走手机热点

---

## 可运行示例（验证与排错）

### 1）连通性验证

客户端连接 WireGuard 后：

```bash
# 内网是否能通
ping 192.168.1.10

# 外网出口是否仍是手机热点（应是你手机运营商/热点出口）
curl ifconfig.me
```

### 2）服务器端检查

```bash
sudo wg
sudo iptables -t nat -S | grep -n MASQUERADE || true
sudo sysctl net.ipv4.ip_forward
```

---

## 解释与原理（为什么这么配）

1) **AllowedIPs 决定路由**：它相当于在客户端路由表里加了一条“更具体”的规则。

2) **为什么要 NAT（MASQUERADE）**：

* 你的公司内网机器（192.168.1.x）通常不知道 `10.200.200.0/24` 这个网段怎么回包
* 通过 NAT，把客户端流量伪装成“来自服务器内网 IP”，就能直接通

3) **为什么需要端口映射**：

* WireGuard 的握手基于 UDP
* 若服务器在 NAT 后面，外网数据包无法被路由器自动转发到它

---

## E — Engineering（工程应用）

### 场景 1：手机热点开发 + 访问内网 Gitlab

**背景**：外网走热点更稳定，但代码仓库只在内网。  
**为什么适用**：Split Tunnel 只接管内网，不影响外网。  

```bash
git clone http://192.168.1.20/your/repo.git
```

### 场景 2：外网排查内网服务（健康检查）

**背景**：需要在外面快速确认某个内网服务是否正常。  
**为什么适用**：VPN 让你像在办公室一样访问内网 IP。  

```bash
curl -sS http://192.168.1.10:8080/health || echo "health check failed"
```

### 场景 3：把 WireGuard 作为“最小权限入口”

**背景**：不想把 SSH/数据库暴露公网。  
**为什么适用**：只开放 WireGuard UDP，一个口进入受控内网。  

```bash
ssh user@192.168.1.30
```

---

## R — Reflection（反思与深入）

### 成本与复杂度

* 部署复杂度：中等（关键在公网入口与路由/NAT）
* 运行开销：低（WireGuard 性能很好）

### 常见失败点（优先排查顺序）

1. **Endpoint 不可达**：没做端口映射 / 被防火墙拦截
2. **AllowedIPs 配错**：写成 `0.0.0.0/0` 导致外网走公司
3. **没开 ip_forward**：客户端能握手但访问内网不通
4. **没做 NAT 或回程路由**：内网回包找不到 `10.200.200.0/24`
5. **DNS 问题**：内网域名无法解析（需要内网 DNS）

### 替代方案（当你做不了端口映射）

* **最省事：Tailscale 子网路由**（同样基于 WireGuard，NAT 穿透更友好）
* **自建：公网 VPS 做中转**（Hub-and-Spoke）
* **运维化：FRP / 反向隧道**

---

## S — Summary（总结）

* 你的目标是“分流”，不是“同时连两张网”。
* WireGuard 的分流关键是 `AllowedIPs`：只写公司内网网段。
* 服务器在 NAT 后面时，最难的是“外网入口”（端口映射或中转）。
* 内网访问不通时，优先查：端口可达 → ip_forward → NAT/回程路由。

推荐延伸阅读：

* WireGuard Quick Start
* Linux `ip route` 与 `iptables` 基础

---

## 常见问题与注意事项

1) **公司网关做不了端口映射怎么办？**  
优先用 Tailscale；如果必须自建，可用公网 VPS 做中转。

2) **Windows 连上 WireGuard 后，WSL 能用吗？**  
WSL1 通常没问题；WSL2 多数情况下也可访问同一路由。
若 WSL2 不通，建议开启 Windows 11 的 mirrored networking，或在 WSL 内单独安装 WireGuard。

3) **能不能把数据库端口暴露给外网？**  
不建议。更安全的做法是只暴露 WireGuard，再在内网访问数据库。

---

## 最佳实践与建议

* 只做 Split Tunnel：`AllowedIPs` 只包含内网网段。
* 最小暴露面：公网只开放 UDP 51820。
* 把“端口映射/防火墙/网卡名”写进运维文档，方便交接。
* 先跑通 IP，再解决 DNS（内网域名需要内网 DNS）。

---

## 小结 / 结论

WireGuard 完全可以实现“外网走手机热点、内网走 VPN”。
但成败关键在于：**服务器是否有公网入口**。
能做端口映射就用标准 WireGuard；做不了就考虑 Tailscale 或中转方案。

---

## 参考与延伸阅读

- https://www.wireguard.com/quickstart/
- https://man7.org/linux/man-pages/man8/ip.8.html
- https://man7.org/linux/man-pages/man8/iptables.8.html

---

## 元信息

- **阅读时长**：约 12 分钟
- **标签**：WireGuard、VPN、Split Tunnel、内网访问
- **SEO 关键词**：WireGuard, Split Tunnel, 内网访问, 端口映射, NAT
- **元描述**：WireGuard Split Tunnel 实战教程，教你手机热点上外网同时访问公司内网，含端口映射与排错清单。

---

## 行动号召（CTA）

把你的内网网段（例如 `192.168.1.0/24`）和服务器网卡名（如 `eno2`）发我，
我可以帮你把配置替换成“可直接复制运行”的最终版，并补一份排错命令清单。
