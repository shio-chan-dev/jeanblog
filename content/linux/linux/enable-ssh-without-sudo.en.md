---
title: "Run SSH Without sudo: User-Level sshd on Linux"
date: 2025-10-24
draft: false
---

Below is a full draft based on your SSH startup and debugging process. It is ready for publication on a technical blog.

---

# Run SSH Without sudo on Linux (User-Level sshd Guide)

**Subtitle / Abstract:**
When you have no root access in a lab or restricted server environment, how do you start SSH and access your account remotely? This guide shows how to run `sshd` in your user directory, enable key login, and connect remotely.

**Reading time:** 10 minutes
**Target readers:** intermediate Linux users, researchers, server users, DevOps learners
**Tags:** SSH, sshd, Linux, remote access, non-root, system config
**SEO keywords:** SSH without root, user-level sshd, openssh config, unprivileged ports, remote login failed

---

## Background and motivation

Many research servers and shared hosts do not grant `sudo`. But we still need:

- remote login
- file upload/download
- access from another machine

By default, `sshd` requires root because it binds port 22 and reads system auth info. However, you can run a **user-level SSH service** in your home directory without changing system config.

---

## Core concepts

| Term | Meaning |
| --- | --- |
| **sshd** | SSH server daemon that accepts connections |
| **user-space sshd** | sshd started by a normal user, no root privileges |
| **HostKey** | key pair used to encrypt SSH connections |
| **AuthorizedKeys** | list of public keys allowed to log in |
| **/etc/shadow** | password hash file; non-root cannot read |

---

## Step-by-step: Start user-level SSH

### Step 1: Prepare config

Create directory:

```bash
mkdir -p ~/.ssh
```

Create config `~/.ssh/ssh_config`:

```bash
Port 2222
ListenAddress 0.0.0.0
HostKey /home/<username>/.ssh/ssh_host_ed25519_key
AuthorizedKeysFile /home/<username>/.ssh/authorized_keys
PasswordAuthentication yes
PubkeyAuthentication yes
ChallengeResponseAuthentication no
PidFile /home/<username>/.ssh/sshd.pid
```

> Note: do not use `~` in paths; OpenSSH will not expand it.

---

### Step 2: Generate host keys

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ssh_host_ed25519_key -N ""
chmod 600 ~/.ssh/ssh_host_ed25519_key
```

---

### Step 3: Start user-level sshd

```bash
/usr/bin/sshd -d -f ~/.ssh/ssh_config
```

If you see:

```
Server listening on 0.0.0.0 port 2222.
```

then it is running. Test locally:

```bash
ssh -p 2222 <username>@localhost
```

---

## Explanation

1. **Why use port 2222?**
   Ports < 1024 are privileged and require root. Use 2222 or 8022 instead.

2. **Why "Could not get shadow information"?**
   Non-root users cannot read `/etc/shadow`, so password auth fails. Use public keys instead.

---

## Use SSH key login (recommended)

1. **Generate local key (no email comment):**

   ```bash
   ssh-keygen -t ed25519 -C "" -f ~/.ssh/id_ed25519_noemail
   ```

2. **Add to authorized keys:**

   ```bash
   cat ~/.ssh/id_ed25519_noemail.pub >> ~/.ssh/authorized_keys
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/authorized_keys
   ```

3. **Test login:**

   ```bash
   ssh -i ~/.ssh/id_ed25519_noemail -p 2222 <username>@localhost
   ```

---

## Allow remote access

1. **Ensure sshd listens on all addresses**

   ```bash
   ss -tlnp | grep 2222
   ```

   If output is `127.0.0.1:2222`, it is local only. Set:

   ```
   ListenAddress 0.0.0.0
   ```

   and restart sshd.

2. **Firewall and NAT**

   - If external access shows "Connection refused", firewall or NAT is blocking.
   - If `localhost` works but public IP fails, open the port or configure forwarding.

3. **Run sshd in background**

   ```bash
   nohup /usr/bin/sshd -f ~/.ssh/ssh_config -E ~/.ssh/sshd.log &
   tail -f ~/.ssh/sshd.log
   ```

---

## Common issues

| Issue | Cause | Fix |
| --- | --- | --- |
| `Permission denied (password)` | cannot read `/etc/shadow` | use key auth |
| `Address already in use` | port in use | kill old process or change port |
| `Bind to port failed` | tried port 22 | use port > 1024 |
| `Connection refused` | firewall / NAT block | check listen address and policies |
| `Could not load host key` | HostKey path wrong | use absolute path and chmod 600 |

---

## Best practices

- Use **ed25519** keys (secure and fast).
- In non-root environments, use **key-only auth**.
- Keep `~/.ssh` at 700 and `authorized_keys` at 600.
- Do not expose your home directory or host keys.
- If remote access is needed, ensure `ListenAddress 0.0.0.0` and open ports.

---

## Summary

This guide shows how to:

1. Start SSH without sudo
2. Enable key auth to avoid `/etc/shadow`
3. Support both local and remote login
4. Debug common errors like "Connection refused"

You can now run your own SSH service under a normal account.

---

## References

- [OpenSSH manual](https://www.openssh.com/manual.html)
- [man sshd_config](https://man.openbsd.org/sshd_config)
- [RFC 4251: The Secure Shell Protocol Architecture](https://www.rfc-editor.org/rfc/rfc4251)
- [Linux file permissions](https://wiki.archlinux.org/title/File_permissions_and_attributes)

---

## Call to Action (CTA)

- Try starting your own user-level sshd using the steps above.
- Save and share this guide for restricted environments.
- Share your SSH deployment pitfalls and fixes.

---

Do you want a Markdown version with syntax highlighting ready for publishing?
