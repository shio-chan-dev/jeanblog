---
title: "Auto-start frp on Ubuntu with systemd"
date: 2025-10-23
draft: false
---

# Auto-start frp on Ubuntu: A Complete Guide

**Subtitle / Abstract**
Use systemd to run frp (Fast Reverse Proxy) as a managed service for stable, secure, and monitored auto-start on boot.

**Reading time**: 8 minutes
**Tags**: frp, intranet tunneling, systemd, auto-start, Linux, Ubuntu
**SEO keywords**: frp auto start, Ubuntu frp config, frpc systemd, frps service, intranet tunneling
**Meta description**: Step-by-step systemd setup for frp (frpc/frps) with config templates and troubleshooting.

---

## Target readers

- Developers deploying frps on cloud servers
- Intermediate Linux users building stable home/office tunnels
- DevOps and self-hosting enthusiasts

---

## Background and motivation

Many developers use **frp** to expose internal services (SSH, web, NAS) to the internet. The problem is that running `./frpc -c frpc.ini` manually is inconvenient and unreliable after reboot.

We want **auto-start on boot + auto-restart on failure + centralized logs**, which is exactly what **systemd** provides.

---

## Core concepts

- **frps / frpc**: server and client binaries for frp
- **systemd**: service manager for modern Linux
- **unit file**: configuration for service startup, dependencies, and restart policy

---

## Step-by-step setup

### 1) Install and place files

```bash
sudo mv frpc /usr/local/bin/
sudo chmod +x /usr/local/bin/frpc
sudo mkdir -p /etc/frp
sudo mv frpc.ini /etc/frp/frpc.ini
```

> For the server side, replace `frpc` with `frps` and `frpc.ini` with `frps.ini`.

---

### 2) (Optional) Create a dedicated user

```bash
sudo useradd --system --no-create-home --shell /sbin/nologin frp
sudo chown -R frp:frp /etc/frp
```

---

### 3) Create a systemd unit

Create `/etc/systemd/system/frpc.service`:

```ini
[Unit]
Description=frp client service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=frp
Group=frp
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.ini
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

---

### 4) Start and enable

```bash
sudo systemctl daemon-reload
sudo systemctl start frpc
sudo systemctl enable frpc
```

---

### 5) Check status and logs

```bash
sudo systemctl status frpc
sudo journalctl -u frpc -f
```

Logs are centralized in the systemd journal for easier troubleshooting.

---

## How it works

- `WantedBy=multi-user.target` ensures auto-start during boot.
- `After=network-online.target` starts only after the network is ready.
- `Restart=on-failure` auto-restarts frpc on unexpected exit.

Compared to `@reboot` cron, systemd gives better dependency control, restart policy, and unified logs.

---

## Common issues and fixes

| Issue | Cause | Fix |
| --- | --- | --- |
| Service fails to start | Config file permission issue | Ensure `/etc/frp/frpc.ini` is readable by user `frp` |
| Network not ready | Missing systemd dependencies | Enable `systemd-networkd-wait-online.service` |
| frp cannot connect | Firewall or security group blocks | Open TCP/UDP ports |
| Service not auto-starting | `enable` not run | `sudo systemctl enable frpc` |

---

## Best practices

- Run with **non-root** user for safety.
- Ship logs to ELK/Promtail if needed.
- Enable token auth or TLS in frp configs.
- For multiple frpc instances, use `frpc@name.service` templates.

---

## Summary

You learned how to:

1. Install and configure frp
2. Create a systemd service
3. Enable auto-start and auto-restart
4. Understand common pitfalls

Once you understand systemd, you can manage any custom daemon the same way.

---

## References

- frp docs: https://github.com/fatedier/frp
- systemd.service: https://www.freedesktop.org/software/systemd/man/systemd.service.html
- Ubuntu Server Guide - systemd: https://ubuntu.com/server/docs/service-systemd

---

## Call to Action

Copy the unit file to your server and run `sudo systemctl enable --now frpc`.
If it works, share what you are exposing via frp. You can also publish a template or script for others.
