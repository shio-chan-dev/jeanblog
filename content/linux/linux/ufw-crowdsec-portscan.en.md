---
title: "UFW + CrowdSec: Stop Malicious Port Scans (From Fail2ban Pain to a Modern Solution)"
date: 2025-11-22T12:00:00+08:00
slug: ufw-crowdsec-portscan
categories: ["Linux", "Security"]
tags: ["CrowdSec", "UFW", "Fail2ban", "FRP", "port-scan", "server-security"]
draft: false
---

# UFW + CrowdSec: Stop Malicious Port Scans

**Subtitle / Abstract:** How do you protect exposed server ports? This guide shows how to move past Fail2ban regex hell and build a stable, automated, intelligent port-scan defense system.

---

## Target readers

- Developers using FRP or reverse tunnels
- Operators of cloud servers (Tencent, Alibaba, AWS, etc.)
- Linux users who want to stop port scans and SSH brute force
- People using Fail2ban who want a modern alternative
- Anyone improving personal server security

---

## Background / Motivation: Why you need port-scan defense

When you run FRP (frps + frpc) or expose multiple ports, you will often see:

- Massive scans: repeated SYN probes
- Malicious connection attempts
- SSH password brute force
- Automated scans of 6001-6010, 7000, 22, 8080, etc.

Traditional approaches have weaknesses:

- UFW only blocks passively
- Fail2ban is regex-heavy, error-prone, and lacks behavior analysis
- FRPS logs are hard to match in Fail2ban
- Attacks still consume frps/sshd resources and can cause slowdowns

We need a modern system: **no regex, auto detection, intelligent IP banning**.

---

## Core concepts

- **FRP (frps / frpc)**: reverse tunnel tool, often exposes many TCP ports (e.g., 6001-6010)
- **UFW**: Ubuntu firewall, but not intelligent
- **Fail2ban**: log-matching ban tool that requires regex
- **CrowdSec (recommended)**: modern open-source IPS that detects port scans and brute force with behavior analysis and low resource usage

---

## Practical guide: Auto-block port scans with CrowdSec (Ubuntu/Debian)

### 1) Install CrowdSec

```bash
curl -s https://packagecloud.io/install/repositories/crowdsec/crowdsec/script.deb.sh | sudo bash
sudo apt install crowdsec -y
```

### 2) Install firewall bouncer (iptables/UFW)

```bash
sudo apt install crowdsec-firewall-bouncer-iptables
```

CrowdSec will manage blocking automatically.

### 3) What it detects out of the box

- TCP port scans
- FRP brute-force attempts
- SSH brute force
- High-rate connections (DoS-like)
- Suspicious sequences (behavior analysis)

No extra rules needed for 6001-6010 and other ports.

### 4) View banned IPs

```bash
sudo cscli decisions list
```

Example:

```
ID   Scope   Value             Reason    Duration
1    Ip      195.24.237.176    portscan  4h
2    Ip      213.199.63.251    ssh-bf    24h
```

### 5) Manual ban (optional)

```bash
sudo cscli decisions add --ip 195.24.237.176
```

### 6) Dashboard (optional)

```bash
sudo apt install crowdsec-lapi
```

---

## Why CrowdSec > Fail2ban

| Feature | Fail2ban | CrowdSec |
| --- | --- | --- |
| Port-scan detection | No | Yes (auto) |
| FRP log support | Regex heavy | No log match needed |
| Config complexity | High | Low |
| Performance | Medium | Very low |
| Extensibility | Weak | Modular + behavior analysis |
| Visualization | None | Dashboard |
| Resource usage | Medium | RAM < 20MB |

CrowdSec is a modern replacement for Fail2ban with lower overhead and stronger detection.

---

## Fail2ban pitfalls (why it fails in FRP scenarios)

- FRPS logs are complex; IP fields shift and are inconsistent
- Regex must be perfect; a small mistake matches nothing
- Logs include colons, brackets, and ports, which break patterns
- Host IP may be internal (e.g., 10.5.100.2), causing mismatched source IPs
- UFW log formats vary; Fail2ban cannot extract IP reliably
- Encoding issues can lead to "No failure-id group"

---

## Risks and notes

1. Blocking can briefly impact FRP or SSH; always keep a backup access method (cloud console).
2. CrowdSec may false-positive crawlers. Whitelist trusted IPs:
   - `sudo cscli machines list`
   - `sudo cscli decisions delete --ip <trusted-ip>`
3. FRP often hides real client IPs in logs; CrowdSec works at the kernel level, so it still sees the source.

---

## Best practices

- Replace Fail2ban with CrowdSec (strongly recommended)
- Close unused FRP ports, use strong tokens and encryption
- Use SSH keys only, disable password auth
- Keep UFW default deny incoming
- Check bans regularly: `cscli decisions list`
- Consider Cloudflare Tunnel as an alternative to FRP

---

## Summary

This guide covered:

- How to detect and block port scans
- Why Fail2ban regex often fails
- Why FRP logs are a poor fit for Fail2ban
- How CrowdSec provides automated, low-maintenance protection

**Final solution: UFW + CrowdSec = stable, automated, low-maintenance server defense.**

---

## References

- CrowdSec docs: https://doc.crowdsec.net
- CrowdSec bouncer: https://github.com/crowdsecurity/cs-firewall-bouncer
- Fail2ban docs: https://fail2ban.readthedocs.io
- FRP: https://github.com/fatedier/frp
- UFW docs: https://wiki.ubuntu.com/UFW
