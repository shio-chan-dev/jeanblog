---
title: "LAN Git Bare on WSL2"
date: 2025-10-22
draft: false
---

# Access a Git Bare Repo on Windows WSL2 from the LAN

In development, you often need to share Git repositories across multiple machines. If you use WSL2 on Windows and want other LAN machines to access a Git bare repo inside WSL2, this guide walks you through the setup.

---

## 1. Create a Git bare repo in WSL2

In WSL2, go to the target directory:

```bash
git init --bare my_project.git
```

- `my_project.git` is a bare repo with no working tree, only Git data.
- A bare repo behaves like a remote and can be cloned and pushed.

---

## 2. Enable SSH in WSL2

Other machines will access via SSH.

1. Install SSH server:

```bash
sudo apt update
sudo apt install openssh-server -y
```

2. Start SSH:

```bash
sudo service ssh start
```

3. Check status:

```bash
sudo service ssh status
```

4. Default port is 22; you can change it in `/etc/ssh/sshd_config`.

---

## 3. Get the WSL2 IP

In WSL2:

```bash
ip addr
```

Find the `inet` under `eth0`, for example:

```
inet 172.25.190.21/20
```

> Note: WSL2 IP can change after reboot.

---

## 4. Configure Windows Firewall

Allow SSH port through firewall:

1. Windows Firewall -> Advanced settings -> Inbound rules -> New rule
2. Rule type: Port -> TCP -> Port 22 (or custom like 2222)
3. Allow connection -> Apply to Domain/Private/Public
4. Name the rule and finish

---

## 5. Recommended: Windows port forwarding

Because WSL2 IP changes, use Windows port forwarding:

1. Open PowerShell (Admin):

```powershell
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=<WSL_IP>
```

2. From another LAN machine, access via Windows IP + 2222:

```bash
git clone ssh://user@WINDOWS_IP:2222/home/user/my_project.git
```

- `user` is your WSL2 username
- `WINDOWS_IP` is the Windows host LAN IP

---

## 6. Clone, push, pull from another machine

Clone:

```bash
git clone ssh://user@WINDOWS_IP:2222/home/user/my_project.git
```

Commit and push:

```bash
git add .
git commit -m "update"
git push origin main  # or master
```

Pull updates:

```bash
git pull origin main
```

---

## 7. Summary

1. WSL2 has a virtual network; its IP may change on each boot.
2. Port forwarding + firewall rules are the most reliable solution.
3. A bare repo inside WSL2 works like a remote for LAN access.

With these steps, multiple LAN machines can access a WSL2 Git repo for easy collaboration.
