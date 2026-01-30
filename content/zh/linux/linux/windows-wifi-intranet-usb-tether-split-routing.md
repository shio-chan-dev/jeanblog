---
title: "Windows 双网络分流：公司 WiFi 走内网，手机 USB 共享走外网"
date: 2026-01-30T08:57:22+08:00
draft: false
categories: ["Linux", "Network"]
tags: ["Windows", "USB 网络共享", "热点", "路由分流", "跃点数", "metric", "route print"]
description: "在 Windows 上同时连接公司 WiFi 和手机 USB 网络共享，通过跃点数（metric）与静态路由实现“内网走公司、外网走手机”。"
keywords: ["Windows 路由分流", "USB tethering", "跃点数", "metric", "route print", "公司内网", "手机热点"]
---

### **标题**

Windows 双网络分流：公司 WiFi 走内网，手机 USB 共享走外网

---

### **副标题 / 摘要**

你想访问公司内网（例如 `192.168.x.x` 的 Gitlab / 内部 API / 数据库），同时让浏览网页、下载等互联网流量走手机热点。
最稳的做法是：**公司 WiFi 只负责内网，手机 USB 网络共享作为默认外网出口**，并用 Windows 的路由优先级（跃点数/metric）把流量分开。

---

### **目标读者**

* 需要访问公司内网服务，但不想让外网走公司出口的开发者/运维
* 经常在公司/外出之间切换网络，希望“插手机就能分流”的同学
* Windows 10/11 用户（公司 WiFi + 手机 USB 共享）

---

### **背景 / 动机**

当你同时连上：

* 公司 WiFi（内网：`192.168.x.x`）
* 手机 USB 网络共享（外网：互联网）

Windows 会出现两个“默认网关”。如果不配置，系统可能随机抢路由，导致：

* 外网有时走公司（慢/受限）
* 内网有时走手机（根本不通）
* VPN/虚拟网卡（如 Tailscale、SSL VPN、WSL）插一条路由就更混乱

这篇文章的目标是把行为固定下来：

* **外网默认走手机**
* **内网稳定走公司 WiFi**

---

### **核心概念**

* **默认路由（Default route）**：`0.0.0.0/0`，没有更具体匹配时走它（通常就是“上网出口”）。
* **最长前缀匹配**：越具体的网段路由优先（例如 `192.168.1.0/24` 会优先于 `0.0.0.0/0`）。
* **跃点数 / Metric**：当有多条可用路径时，Windows 选择 **metric 更小** 的那条。
* **接口跃点数（Interface metric）**：网卡层面的优先级，影响默认路由选择。
* **静态路由（Static route）**：手动指定某个网段必须走哪个网关（可选但更稳）。

---

## A — Algorithm（题目与算法）

### 题目还原

> 电脑连接公司 WiFi（只负责访问公司内网），手机用 USB 网络共享（只负责上互联网），怎么配置才能稳定分流？

### 核心策略

1. 让手机 USB 网卡成为默认路由（metric 更小）
2. 保持公司 WiFi 的内网路由可用（`192.168.x.x` 走 WiFi）
3. 如有 VPN 抢内网网段，关闭 VPN 或用静态路由盖住它

---

## C — Concepts（核心思想）

### Windows 并不会“先走手机，失败再换 WiFi”

很多人直觉会以为：访问内网如果走手机走不通，Windows 会自动切回公司 WiFi。
实际上不是——Windows 发包时会**直接查路由表**，一次选定路径，不会“试错切换”。

路由选择规则可以记成一句话：

> **先匹配更具体的网段，再在候选路径里选 metric 更小的。**

### 你要分流的本质

把流量分成两类：

* **内网流量**：`192.168.0.0/16` 或 `192.168.1.0/24`（以你公司实际网段为准）
* **外网流量**：除内网外的一切（最终走 `0.0.0.0/0` 默认路由）

---

## 实践指南 / 步骤（Windows 10/11）

> 以下以公司内网 `192.168.1.0/24` 举例，若你公司是 `10.0.0.0/8` 或 `192.168.0.0/16`，替换网段即可。

### Step 0：准备与注意事项

* 手机要开数据流量，并开启 **USB 网络共享（USB tethering）**
* 如果你开了 VPN（例如 Tailscale/SSL VPN），先暂时关闭，避免它插入内网路由（后面有处理方法）
* 若公司有安全规定（禁止外网共享/双网卡），先遵守公司政策

---

### Step 1：同时连接两张网络

1) 连接公司 WiFi（用于内网）  
2) 手机插电脑，打开 USB 网络共享（用于外网）

你在 Windows 的 `ncpa.cpl`（网络连接）里通常会看到：

* Wi-Fi / WLAN（公司 WiFi）
* 以太网 X（Remote NDIS based Internet Sharing Device）（手机 USB 网卡）

---

### Step 2：确认两张网卡的 IP 与网关（只看关键行）

打开 PowerShell / CMD：

```bat
ipconfig
```

你需要确认两张网卡各自的：

* IPv4 地址
* 默认网关

示例（仅示意，数值以你机器为准）：

```
Wi-Fi:
  IPv4 Address . . . . . . . . . . : 192.168.1.7
  Default Gateway . . . . . . . . : 192.168.1.1

USB Ethernet (Remote NDIS):
  IPv4 Address . . . . . . . . . . : 192.168.232.75
  Default Gateway . . . . . . . . : 192.168.232.242
```

---

### Step 3：设置“手机 USB 网卡”为默认出口（metric 更小）

#### 方法 A（推荐）：图形界面设置接口跃点数

1) `Win + R` 输入 `ncpa.cpl`  
2) 找到手机网卡（Remote NDIS）→ 右键 **属性**  
3) 双击 **Internet 协议版本 4 (TCP/IPv4)** → **高级**  
4) 取消勾选 **自动跃点数**  
5) **接口跃点数** 填 `10`（或更小，比如 5/10）

#### 方法 B：PowerShell 一条命令（可选）

先查看接口名：

```powershell
Get-NetIPInterface -AddressFamily IPv4 | Sort-Object InterfaceMetric | Format-Table ifIndex,InterfaceAlias,InterfaceMetric
```

再设置（把接口名替换成你的 Remote NDIS 对应名称）：

```powershell
Set-NetIPInterface -InterfaceAlias \"以太网 3\" -InterfaceMetric 10
```

---

### Step 4：降低公司 WiFi 的默认优先级（metric 更大）

同样在 Wi-Fi 网卡的 IPv4 高级设置里：

* 取消自动跃点数
* 设置接口跃点数：`50`（或更大，比如 50/100）

这一步的目标是：**即使 Wi-Fi 也能上网，也不要抢默认外网出口**。

---

### Step 5：验证分流是否生效（必须做）

#### 1）看默认路由是不是手机（metric 最小）

```bat
route print
```

你应该能看到两条默认路由，但手机那条 metric 更小：

```
0.0.0.0    0.0.0.0    <PHONE_GW>   <PHONE_IF_IP>   10
0.0.0.0    0.0.0.0    <WIFI_GW>    <WIFI_IF_IP>    50
```

#### 2）测试内网能通

```bat
ping <INTRANET_SERVICE_IP>
```

例如：

```bat
ping 192.168.1.10
```

#### 3）确认外网出口是手机

浏览器打开 `https://ipinfo.io` 或在 PowerShell：

```powershell
curl ifconfig.me
```

显示的公网 IP 应该是你的手机运营商出口（而不是公司出口）。

---

### Step 6（建议）：为公司内网网段加一条“强制路由”（更稳）

如果你环境里有 VPN/虚拟网卡，或者 Windows 仍偶发走错路由，可以加静态路由“钉死”内网走 WiFi 网关。

例如公司网段是 `192.168.1.0/24`，网关是 `192.168.1.1`：

```bat
route -p add 192.168.1.0 mask 255.255.255.0 192.168.1.1 metric 1
```

说明：

* `-p` 表示永久生效（重启不丢）
* `metric 1` 让它优先级最高

撤销命令：

```bat
route delete 192.168.1.0
```

如果你公司内网更大（例如 `192.168.0.0/16`），可以改成：

```bat
route -p add 192.168.0.0 mask 255.255.0.0 192.168.1.1 metric 1
```

> 注意：网段要以公司实际为准，避免把不该走内网的流量也导进去。

---

### Step 7：处理 VPN 抢路由（以 Tailscale 为例）

如果你开着 Tailscale 并启用了子网路由/出口节点，它可能会插入类似路由：

```
192.168.1.0/24 -> 100.100.100.100 (metric 很小)
```

这会导致你的内网流量优先走 VPN，而不是走公司 Wi-Fi。

解决思路：

* 不需要 VPN：直接关闭

```powershell
tailscale down
```

* 需要 VPN 但不想抢这段路由：取消子网路由/exit node（按你实际配置调整）

如果你必须同时开 VPN + 走公司 WiFi 内网，优先用 **Step 6 的静态路由** 把内网钉回 WiFi。

---

### Step 8：WSL 能不能用？（Windows + WSL2）

多数情况下，Windows 配好分流后：

* Windows 本机访问内网 ✅
* WSL2 访问内网 ✅（因为流量最终也从 Windows 出去）

如果 WSL2 偶发解析内网域名失败，先排查 DNS（见 FAQ）。

---

## 可运行示例（一套“验证脚本”）

你可以把下面这组命令当作验收清单：

```bat
route print
ping 192.168.1.10
tracert 192.168.1.10
tracert 8.8.8.8
```

你预期看到的现象：

* `tracert 192.168.1.10` 的第一跳应是公司网关（WiFi）
* `tracert 8.8.8.8` 的第一跳应是手机网关（USB）

---

## 解释与原理（为什么这么做）

1) **内网路由为什么不用 metric 也能生效？**  
因为 `192.168.1.0/24` 比 `0.0.0.0/0` 更具体，按最长前缀匹配会优先命中内网段。

2) **默认路由为什么靠 metric 决定？**  
因为公司 WiFi 和手机 USB 都可能提供 `0.0.0.0/0`，这时就按 metric 选更小的那个作为“默认上网出口”。

3) **为什么要加静态路由？**  
当 VPN/虚拟网卡插入更低 metric 的内网路由时（例如 `192.168.1.0/24 -> VPN`），静态路由可以“盖住”干扰，保证内网始终走公司 WiFi。

---

## 常见问题与注意事项

1) **必须开手机热点吗？**  
你用的是 USB 网络共享（tethering）。只要手机在共享网络给电脑，就等同于“开热点”；不一定要开 WiFi 热点。

2) **我只设置了 metric，内网域名（例如 gitlab.company）解析失败？**  
常见原因是 DNS 走了手机的 DNS。解决方法（按推荐顺序）：

- 直接用内网 IP 访问（最快验证）
- 给 Wi-Fi 网卡设置公司 DNS（适合长期使用）
- 必要时用 `hosts` 固定少量内网域名

如果你愿意贴一下 `ipconfig /all` 中的 DNS 服务器，我可以帮你判断该改哪张网卡。

3) **我已经能分流了，但偶尔内网还是走错？**  
加 Step 6 的静态路由，或关闭 VPN/虚拟网卡的子网路由功能。

4) **为什么不推荐“同时连两个 WiFi”？**  
多数电脑只有一张无线网卡，稳定性差；更推荐 WiFi + USB/网线的双网卡组合。

5) **会不会造成安全风险？**  
双网络环境可能被公司视为高风险（内外网桥接）。务必遵守公司安全政策，不要开启网络共享/桥接功能。

---

## 最佳实践与建议

* 手机 USB（外网）metric 设小（例如 10），公司 WiFi metric 设大（例如 50）
* 用 `route print` 做验收：默认路由走手机，内网网段走 WiFi
* 有 VPN 时先 `tailscale down` / 关闭子网路由，避免抢内网段
* 需要长期稳定：为公司内网网段加一条 `route -p add ...` 静态路由

---

## 小结 / 结论

要实现“公司 WiFi 只访问内网、手机 USB 只负责外网”，关键不是“让系统试错”，而是**把路由规则写清楚**：

* 默认路由（`0.0.0.0/0`）用 metric 指向手机 USB
* 公司内网网段用更具体路由（必要时静态路由）指向 WiFi

配置完成后，你就能做到：外网稳定走手机，内网稳定走公司 WiFi，互不干扰。

---

## 参考与延伸阅读

- Windows `route` 命令：https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/route_ws2008
- PowerShell `Set-NetIPInterface`：https://learn.microsoft.com/en-us/powershell/module/nettcpip/set-netipinterface

---

## 元信息

* **阅读时长**：约 8–10 分钟
* **标签**：Windows、路由分流、USB 网络共享、metric
* **SEO 关键词**：Windows 路由分流, USB tethering, 跃点数, route print
* **元描述**：Windows 同时连公司 WiFi 与手机 USB 网络共享，通过 metric 与静态路由实现内网走公司、外网走手机。

---

## 行动号召（CTA）

如果你愿意，把你的 `ipconfig` 里两张网卡的 **IPv4 + 默认网关 + DNS**（末段可打码）贴出来，
我可以帮你确认：metric 是否合理、静态路由该加哪条，以及是否存在 VPN 抢路由的情况。
