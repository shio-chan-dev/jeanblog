---
title: "Run sshd Without sudo: Troubleshooting and Persistent User-Level SSH"
date: 2025-10-24
draft: false
---

**Title:**
Run sshd Without sudo: Troubleshooting, nohup, and systemd (User-Level SSH)

**Subtitle / Abstract:**
How to run OpenSSH as a normal user, solve common errors like "connection refused", "password auth failed", and `start-limit-hit`, and keep sshd alive using nohup or systemd.

**Target readers:**
Intermediate Linux users, researchers on shared servers, and anyone who needs SSH without root.

---

## 1. Background / Motivation

In some lab or shared environments, regular users do not have sudo. The default sshd service cannot be started. If you need to:

- remote into your Linux host
- use VS Code Remote or SCP
- but cannot change system config

then you must run sshd in **user space**. This introduces issues: port conflicts, firewall rules, auth failures, and `start-limit-hit`.

---

## 2. Core concepts

| Term | Meaning |
| --- | --- |
| **sshd** | OpenSSH daemon that handles SSH logins |
| **user-level sshd** | sshd started by a normal user, no root privileges |
| **authorized_keys** | list of allowed public keys |
| **nohup** | run a process detached from the terminal |
| **systemd --user** | user-level systemd instance for services |
| **start-limit-hit** | systemd pauses restarts after frequent failures |

---

## 3. Full setup steps

### 1) Generate and configure SSH keys

```bash
ssh-keygen -t ed25519 -C "" -f ~/.ssh/id_ed25519_noemail
cat ~/.ssh/id_ed25519_noemail.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Ensure `~/.ssh/authorized_keys` permissions are correct.

---

### 2) Create a user-level sshd config

`~/.ssh/ssh_config_pub`

```bash
Port 2223
ListenAddress 0.0.0.0
HostKey /home/chenhm/.ssh/ssh_host_ed25519_key
AuthorizedKeysFile /home/chenhm/.ssh/authorized_keys
PasswordAuthentication no
PubkeyAuthentication yes
PidFile /home/chenhm/.ssh/sshd_pub.pid
LogLevel INFO
SyslogFacility AUTH
```

Generate host key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ssh_host_ed25519_key -N ""
```

---

### 3) Start in debug mode

```bash
/usr/bin/sshd -d -f ~/.ssh/ssh_config_pub
```

If you see:

`Server listening on 0.0.0.0 port 2223`

then it is running.

---

## 4. Two ways to keep it running

### Option A: nohup (simplest)

```bash
nohup /usr/bin/sshd -f ~/.ssh/ssh_config_pub -E ~/.ssh/sshd_pub.log >/dev/null 2>&1 &
```

- Runs after terminal closes
- Check process:

  ```bash
  ps -ef | grep "sshd -f"
  ```
- Check logs:

  ```bash
  tail -f ~/.ssh/sshd_pub.log
  ```
- Stop:

  ```bash
  pkill -f "sshd -f /home/chenhm/.ssh/ssh_config_pub"
  ```

Pros: no dependencies, works instantly.
Cons: does not auto-start after reboot.

---

### Option B: systemd user service (auto-restart/auto-start)

#### 1) Create the unit file

`~/.config/systemd/user/sshd-user.service`

```ini
[Unit]
Description=User-level SSH server

[Service]
Type=forking
ExecStart=/usr/bin/sshd -f /home/chenhm/.ssh/ssh_config_pub -E /home/chenhm/.ssh/sshd_pub.log
PIDFile=/home/chenhm/.ssh/sshd_pub.pid
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

#### 2) Enable and start

```bash
systemctl --user daemon-reload
systemctl --user enable sshd-user
systemctl --user start sshd-user
```

#### 3) Verify

```bash
systemctl --user status sshd-user
ss -tlnp | grep sshd
```

You should see `Active: active (running)` and `0.0.0.0:2223`.

---

## 5. Troubleshooting table

| Error | Cause | Fix |
| --- | --- | --- |
| `Connection refused` | sshd not listening on public interface or firewall blocked | set `ListenAddress 0.0.0.0`, check `ss -tlnp` |
| `Permission denied (password)` | no access to `/etc/shadow` | use public key auth |
| `Bind to port ... failed: Address already in use` | port already used by old sshd | `pkill -f "sshd -f"` |
| `start-limit-hit` | systemd sees frequent crashes | set `Type=forking` and `PIDFile=` |
| No logs | wrong path or permission | use `-E ~/.ssh/sshd.log` |

---

## 6. Why this works

- User-level sshd does not need root because it binds to ports >= 1024.
- Public key auth avoids `/etc/shadow` access.
- `Type=forking` lets systemd track the daemon correctly.
- `PIDFile` helps systemd manage the process.

---

## 7. Notes

1. **Port > 1024**: non-root cannot bind to privileged ports.
2. **Firewall**: must allow your chosen port.
3. **Permissions**: `~/.ssh` must be 700 and `authorized_keys` must be 600.
4. **Multiple instances**: use separate `PidFile` and log paths.
5. **Auto-start**: `systemctl --user enable sshd-user`.

---

## 8. Best practices

- Use **nohup** for testing or temporary runs.
- Use **systemd --user** for stable long-term service.
- Expose only key-based auth on public interfaces.
- Separate internal vs external ports.
- Use `@reboot` cron as fallback if systemd is unavailable.

---

## 9. Conclusion

This guide showed how to deploy SSH without sudo:

1. Generate keys and enable key auth
2. Create user-level sshd config
3. Validate with nohup, then stabilize with systemd
4. Fix `start-limit-hit`, port conflicts, and auth failures

You get:

- Multiple ports and instances
- Auto-restart
- Auto-start
- Secure remote access

---

## References

- [OpenSSH manual](https://man.openbsd.org/sshd.8)
- [systemd user services](https://wiki.archlinux.org/title/Systemd/User)
- [OpenSSH key management](https://www.ssh.com/academy/ssh/keygen)

---

**Meta**

- Reading time: about 10 minutes
- Tags: `SSH`, `Linux`, `systemd`, `nohup`, `no-sudo`
- SEO keywords: `no-sudo sshd systemd user OpenSSH start-limit-hit`
- Meta description: A complete guide to running OpenSSH without sudo and fixing common startup errors.

---

**Call to Action (CTA)**
Try running a user-level sshd on your lab server. If this helps, share your setup and lessons learned.
