---
title: "Ping Works but SSH Fails: A Real Case of SSH vs VNC"
date: 2025-10-24
draft: false
---

# Ping Works but SSH Fails: A Real Case of SSH vs VNC

> **Subtitle:** From connection refusal to protocol identification: understand TCP, SSH, and VNC
> **Reading time:** 7 minutes
> **Tags:** network troubleshooting, SSH, VNC, Linux, remote access
> **SEO keywords:** SSH connection failed, kex_exchange_identification, VNC port 5905, RFB 003.008, SSH vs VNC

---

## Target readers

- Linux users, developers, and server admins
- Engineers learning systematic network troubleshooting
- Readers interested in SSH/VNC protocol behavior

---

## Background and motivation

Have you seen this?

> "The server can be pinged, but SSH does not connect."

This is common on hosts running multiple services (SSH, VNC, HTTP). This article walks through a real case: from "SSH failed" to finding out the port was actually VNC.

---

## Symptoms

Command:

```bash
ssh chenhm@101.6.142.82 -p 5905
```

Output:

```
kex_exchange_identification: Connection closed by remote host
Connection closed by 101.6.142.82 port 5905
```

Ping test:

```bash
ping 101.6.142.82
```

It succeeds with no packet loss.

So we know:

- Host is online
- Network is reachable
- SSH handshake failed

---

## Core concepts

| Concept | Meaning |
| --- | --- |
| **Ping** | ICMP test for connectivity only |
| **TCP** | Transport protocol that builds connections |
| **SSH** | Application protocol on top of TCP for secure login |
| **VNC / RFB** | Remote desktop protocol (Remote Frame Buffer) |

In short: **Ping OK != SSH OK** because they are different layers.

---

## Troubleshooting steps

### Step 1. Test TCP connectivity

```bash
telnet 101.6.142.82 5905
```

Output:

```
Trying 101.6.142.82...
Connected to 101.6.142.82.
Escape character is '^]'.
RFB 003.008
```

Key clue: `RFB 003.008` is the **VNC handshake string** (Remote Frame Buffer v3.8).

This means:

- Port 5905 is open
- It is running **VNC**, not SSH

---

## Why this happens

After TCP connects, SSH sends a greeting like `SSH-2.0-OpenSSH_8.x`.
A VNC server replies with `RFB 003.008` instead.
Protocol mismatch causes the SSH client to close, resulting in `kex_exchange_identification`.

---

## Verification

1) **Check the process on the port**

```bash
sudo ss -tlnp | grep 5905
```

Possible output:

```
LISTEN  0  5  0.0.0.0:5905  ...  /usr/bin/Xvnc
```

2) **Check SSH port**

```bash
sudo grep ^Port /etc/ssh/sshd_config
```

If it returns `Port 22`, SSH is still on the default port.

---

## Correct connection

### If you want the GUI

Use a VNC client:

```bash
vncviewer 101.6.142.82:5905
```

Or tools like:

- RealVNC
- TigerVNC
- TightVNC

### If you want the terminal

Use SSH on the correct port:

```bash
ssh chenhm@101.6.142.82 -p 22
```

---

## Common issues and fixes

| Problem | Cause | Fix |
| --- | --- | --- |
| `Connection closed by remote host` | Protocol mismatch (SSH to VNC) | Use the correct protocol |
| SSH fails on all ports | SSH service not running | `sudo systemctl start sshd` |
| VNC refused | Firewall blocked | `firewall-cmd --add-port=5905/tcp --permanent` |
| SSH disconnected | fail2ban ban | check `/var/log/auth.log` |

---

## Best practices

- **Separate port and protocol**: port number alone does not identify service type.
- **Use `telnet` or `nc`** to read protocol banners.
- **Check logs**: `journalctl -u ssh`, `/var/log/auth.log`.
- **Define a clear port map** for multi-service hosts:

```
SSH -> 22
VNC -> 5900+
HTTP -> 80/8080
HTTPS -> 443
```

---

## Summary

This case shows:

1. How to separate network, transport, and application layer issues
2. How to identify protocol banners (RFB vs SSH)
3. How to find the real service on a port

One-line conclusion:

**SSH is fine; you connected to the wrong service.**

---

## References

- [OpenSSH Manual](https://www.openssh.com/manual.html)
- [TigerVNC GitHub](https://github.com/TigerVNC/tigervnc)
- [Linux man pages: ssh, telnet, ss, netstat]
- [RFC 6143: The Remote Framebuffer Protocol (RFB)](https://datatracker.ietf.org/doc/html/rfc6143)

---

## Call to Action

Try this on your own server:

```bash
nc <server_ip> <port>
```

Check the banner you get back. You may discover more "hidden services".
