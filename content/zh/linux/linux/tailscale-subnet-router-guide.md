---
title: "Tailscale 子网路由实战：外网访问公司内网的最稳方案"
date: 2026-01-29T16:13:14+08:00
draft: false
categories: ["linux", "network"]
tags: ["tailscale", "vpn", "子网路由", "内网访问", "split-tunnel"]
description: "一篇可直接落地的 Tailscale 子网路由教程，解决无公网 IP 情况下的内网访问与分流。"
keywords: ["Tailscale", "Subnet Router", "Split Tunnel", "内网访问", "NAT 穿透"]
---

### **标题**

Tailscale 子网路由实战：外网访问公司内网的最稳方案

---

### **副标题 / 摘要**

当公司内网没有公网 IP 时，Tailscale 的子网路由是最省事、最稳定的方案。
本文给出完整的部署步骤、验证方法与排错清单，适合直接落地。

---

### **目标读者**

* 需要在外网访问公司内网服务的开发者/运维
* 公司没有官方 VPN，且内网在 NAT 后面
* 希望实现“外网走手机热点、内网走 VPN”的分流需求

---

### **背景 / 动机**

很多公司内网服务器只有 192.168.x.x 等私有地址，
外网无法直接访问，传统 WireGuard 需要公网 IP 或端口映射。
Tailscale 基于 WireGuard，但可以穿透 NAT，
**无需公网入口**即可实现安全访问，是更工程化的解法。

---

### **核心概念**

* **子网路由（Subnet Router）**：让一台内网机器“代理”整个内网段
* **Split Tunnel（分流）**：只有内网流量走 VPN，外网流量不变
* **Advertise Routes**：向 tailnet 宣告你能到达的内网网段
* **Approve Routes**：在控制台批准路由才能生效

---

## 实践指南 / 步骤（完整落地）

以下以公司内网 `192.168.1.0/24` 为例，你可替换成自己的网段。

### 1）在公司内网服务器安装 Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

登录后，服务器会出现在控制台设备列表中。

### 2）将服务器设置为子网路由

```bash
sudo tailscale up --advertise-routes=192.168.1.0/24
```

### 3）到控制台批准路由（必须）

打开控制台：

```
https://login.tailscale.com/admin/routes
```

看到 `192.168.1.0/24` 后点击 **Approve/Enable**。

### 4）开启 IP 转发（必须）

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo sysctl -w net.ipv6.conf.all.forwarding=1
```

永久生效：

```bash
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv6.conf.all.forwarding=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 5）如内网不通，再补 NAT（按需）

```bash
sudo iptables -t nat -A POSTROUTING -o <内网网卡> -j MASQUERADE
```

> `<内网网卡>` 通常是 `eno2`/`eth0`/`ens*`，用 `ip a` 查看。

### 6）在外网电脑安装并登录 Tailscale

Windows / macOS / Linux 安装后登录同一账号即可。

### 7）测试访问内网

```bash
ping 192.168.1.10
curl http://192.168.1.10:8080
```

能通说明子网路由已生效。

---

## 可运行示例（最小验证）

服务器端检查路由是否生效：

```bash
tailscale status
tailscale ip -4
```

客户端快速验证：

```bash
ping 192.168.1.10
```

---

## 解释与原理（为什么这么做）

* Tailscale 通过 NAT 穿透建立点对点通道
* 子网路由让一台内网服务器“转发”整个内网段
* Split Tunnel 只接管内网路由，不影响外网流量

所以你可以：

* **外网走手机热点**
* **内网走 Tailscale**

---

## 常见问题与注意事项

1. **提示 IPv6 forwarding disabled？**  
   开启 `net.ipv6.conf.all.forwarding=1` 即可。

2. **Approve 路由后仍访问不了？**  
   多数是未开启 IPv4 转发或 NAT 未加。

3. **内网域名解析失败？**  
   需要设置公司内网 DNS，或在 Tailscale 管理台配置 DNS。

4. **公司是否允许自建 VPN？**  
   先确认安全政策，避免违规。

---

## 最佳实践与建议

* 子网路由机器选择 **稳定在线** 的服务器
* 内网网段尽量精确（/24 优于 /16）
* 记录路由变更，便于排错
* 如内网多网段，可分批 advertise 并逐一批准

---

## 小结 / 结论

在没有公网 IP 的情况下，Tailscale 子网路由是最稳、最省事的内网访问方案。
只要完成：**安装 → advertise → approve → 转发**，就可以实现外网访问内网。

---

## 参考与延伸阅读

- https://tailscale.com/kb/1019/subnets
- https://tailscale.com/kb/1104/enable-ip-forwarding
- https://tailscale.com/kb/1114/clients

---

## 元信息

- **阅读时长**：约 8 分钟
- **标签**：Tailscale、内网、子网路由、分流
- **SEO 关键词**：Tailscale, Subnet Router, Split Tunnel, 内网访问
- **元描述**：无公网 IP 环境下使用 Tailscale 子网路由访问公司内网的完整实战教程。

---

## 行动号召（CTA）

如果你愿意提供内网网段和系统类型，我可以给你一份可直接复制的配置模板。
