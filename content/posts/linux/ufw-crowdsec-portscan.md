---
title: "用 UFW + CrowdSec，彻底阻止恶意端口扫描：从 Fail2ban 踩坑到终极解决方案"
date: 2025-11-22T12:00:00+08:00
slug: ufw-crowdsec-portscan
categories: ["Linux", "Security"]
tags: ["CrowdSec", "UFW", "Fail2ban", "FRP", "端口扫描", "服务器安全"]
draft: false
---

# 🛡️ 用 UFW + CrowdSec，彻底阻止恶意端口扫描

**副标题 / 摘要：** 如何安全防护你的服务器暴露端口？本文带你从 Fail2ban 的正则地狱走出，构建一个稳定、自动化、智能化的端口扫描防御系统。

---

## 🎯 目标读者

- 使用 FRP / 内网穿透的开发者
- 管理云服务器（腾讯云、阿里云、AWS 等）的运维人员
- 想防御端口扫描、SSH 暴力破解的新手或中级 Linux 用户
- 对 Fail2ban 感兴趣、想升级到更现代安全体系的人
- 想完善服务器安全方案的个人开发者

---

## 💢 背景 / 动机：为什么需要端口扫描防护？

在运行 FRP（frps + frpc）或开放多个端口时，你的服务器通常会遭遇：

- 海量扫描：每秒多次 SYN 探测
- 恶意连接尝试：get a user connection [...]
- SSH 密码爆破
- 自动化脚本扫描 6001–6010、7000、22、8080 等常见端口

传统做法存在痛点：

- 防火墙（UFW）只能被动拒绝
- Fail2ban 配置复杂、依赖正则、容易误判、不支持高级行为分析
- FRPS 日志格式特殊，Fail2ban 很难匹配
- 攻击会占用 frps/sshd 资源，最终导致卡顿、断流

因此，我们需要一个**无需写正则、能自动检测扫描、智能封禁恶意 IP 的现代防御体系**。

---

## 📘 核心概念

- **FRP（frps / frpc）**：用于内网穿透，常暴露大量 TCP 端口（如 6001–6010），容易被扫描。
- **UFW（Uncomplicated Firewall）**：Ubuntu 默认防火墙，但缺乏智能检测功能。
- **Fail2ban**：传统日志匹配型封禁工具，需要手写正则，踩坑概率高。
- **CrowdSec（推荐）**：新一代开放式入侵防御系统 (IPS)，自动检测端口扫描和暴力破解，事件驱动 + 行为分析，资源消耗极低，是 Fail2ban 的现代替代。

---

## 🛠 实践指南：使用 CrowdSec 自动阻止端口扫描（Ubuntu/Debian）

### 1) 安装 CrowdSec

```bash
curl -s https://packagecloud.io/install/repositories/crowdsec/crowdsec/script.deb.sh | sudo bash
sudo apt install crowdsec -y
```

### 2) 安装防火墙封禁组件（iptables / ufw 自动配合）

```bash
sudo apt install crowdsec-firewall-bouncer-iptables
```

CrowdSec 会自动接管封禁动作。

### 3) 自动检测的行为

无需额外配置即可识别：

- TCP 端口扫描
- FRP 暴力连接
- SSH 爆破
- 大量连接（DoS-like）
- 异常行为序列（行为/AI 分析）

无需为 6001–6010 等端口写任何规则。

### 4) 查看被封禁的攻击者

```bash
sudo cscli decisions list
```

示例输出：

```
ID   Scope   Value             Reason    Duration
1    Ip      195.24.237.176    portscan  4h
2    Ip      213.199.63.251    ssh-bf    24h
```

### 5) 手动封禁恶意 IP（可选）

```bash
sudo cscli decisions add --ip 195.24.237.176
```

### 6) Dashboard（可选）

```bash
sudo apt install crowdsec-lapi
```

安装后可直观看到攻击图表和趋势。

---

## 🔍 原理与对比：为什么 CrowdSec > Fail2ban？

| 对比项 | Fail2ban | CrowdSec |
| --- | --- | --- |
| 端口扫描检测 | ❌ 基本不支持 | ⭐ 自动识别 |
| FRP 日志支持 | ❌ 需要复杂正则 | ⭐ 无需日志匹配 |
| 配置复杂度 | 高 | ⭐ 极低 |
| 性能 | 中等 | ⭐ 极低 |
| 能力扩展 | 弱 | ⭐ 模块化、行为分析 |
| 可视化 | 无 | ⭐ 有 Dashboard |
| 资源占用 | 中 | ⭐ RAM < 20MB |

CrowdSec 更像是「Fail2ban 的现代化升级版」，并且资源占用小。

---

## ❓ Fail2ban 踩坑实录（常见失败原因）

- FRPS 日志格式复杂，字段和 IP 位置不固定
- 正则必须 100% 精确，末尾 ^$ 容易导致永不匹配
- 日志中混有冒号、括号、端口号，匹配极难
- 主机地址是内网 IP（如 10.5.100.2），多网卡/转发导致源 IP 不一致
- UFW 输出格式不统一，Fail2ban 无法从内核日志提取 host
- BOM / CRLF 或其他编码问题导致 “No failure-id group”

这些都是 Fail2ban 的常见陷阱，也解释了为何在 FRP/多端口场景中很难成功。

---

## ⚠️ 风险与注意事项

1. 防火墙封禁可能短暂影响 FRP 或 SSH，务必确保有备用登录方式（如云厂商 Web 控制台）。
2. CrowdSec 默认封禁端口扫描，可能误报爬虫，可信 IP 需加入白名单：
   - `sudo cscli machines list`
   - `sudo cscli decisions delete --ip <可信IP>`
3. FRP 常不保留真实客户端 IP，但 CrowdSec 直接在内核网络层捕获连接，可绕过应用层日志缺失。

---

## 🌟 最佳实践清单

- 用 CrowdSec 替代 Fail2ban（强烈推荐）
- 关闭不必要的 FRP 端口，设置强 token 与加密
- SSH 使用密钥登录，禁用密码
- UFW 维持默认 deny incoming
- 定期检查封禁记录：`cscli decisions list`
- 如果合适，考虑用 Cloudflare Tunnel 替代 FRP 暴露

---

## 📘 小结

本文完整经历了：

- 如何识别和阻断端口扫描
- Fail2ban 正则配置失败的原因与坑
- FRP 日志不适合被 Fail2ban 直接解析，UFW 日志匹配困难
- 使用 CrowdSec 实现自动化、高可靠、无需正则的防御体系

**最终方案：UFW + CrowdSec = 稳定、自动化、零维护的服务器入侵防御系统。**

---

## 🔗 参考与延伸阅读

- CrowdSec 官方文档：https://doc.crowdsec.net
- CrowdSec Bouncer：https://github.com/crowdsecurity/cs-firewall-bouncer
- Fail2ban 文档：https://fail2ban.readthedocs.io
- FRP 项目：https://github.com/fatedier/frp
- UFW 文档：https://wiki.ubuntu.com/UFW
