---
title: "How to Set Up Gitea"
date: 2025-10-28
draft: false
---

# Run Gitea Locally: Your Private GitHub (with Existing Repo Import)

**Subtitle / Abstract:**
This guide walks you through installing the lightweight Git server Gitea on your local machine. No root required, no system pollution. Manage, browse, and push projects like GitHub, and import existing repos.

**Target readers:**
Personal developers, indie engineers, and small team leads with basic Git knowledge.

---

## Background / Motivation

Many developers want:

- to host code inside a company or LAN
- to avoid cloud platforms (GitHub/Gitee)
- to have a web UI, pull requests, and code browsing

GitLab is heavy (often multiple GB of RAM). Gitea is:

- lightweight
- a single binary
- supports PR, Wiki, Issues, CI/CD

In minutes, you get a private "mini GitHub".

---

## Core concepts

| Term | Description |
| --- | --- |
| **GitLab** | most powerful open-source Git platform, heavy resource usage |
| **Gitea** | lightweight self-hosted Git service with GitHub-like UI |
| **Bare repo** | repo with history only, no working tree |
| **Pull Request** | merge request from one branch to another |
| **SQLite** | default lightweight database for Gitea |

---

## Setup steps

### 1) Prepare environment

Supported OS: Linux / macOS / Windows
Recommended: RAM >= 512MB, disk >= 1GB

### 2) Create directory and download

```bash
mkdir -p ~/gitea
cd ~/gitea
wget -O gitea https://dl.gitea.io/gitea/1.22.0/gitea-1.22.0-linux-amd64
chmod +x gitea
```

### 3) Start Gitea

```bash
./gitea web --port 3000
```

Open: http://localhost:3000

### 4) Install wizard

Fill in:

- DB type: `SQLite3`
- Repo root: `/home/<username>/gitea/repos`
- Base URL: `http://localhost:3000`
- Create admin account

---

## Runnable example: push an existing repo

Assume your local project is `/home/gong/projects/scrapy`:

1) Create a repo named `scrapy` in Gitea
2) In your project directory:

```bash
cd ~/projects/scrapy
git remote set-url origin http://localhost:3000/JeanphiloGong/scrapy.git
git push -u origin --all
git push -u origin --tags
```

Refresh the web UI to see full history.

---

## Register as a system service

### 1) Prerequisites

Assume Gitea is installed at:

```
/home/gong/gitea
```

Binary path:

```
/home/gong/gitea/gitea
```

Run as user `gong` and do not use root.

---

### 2) Create systemd service

Create `/etc/systemd/system/gitea.service`:

```ini
[Unit]
Description=Gitea (Self-hosted Git Service)
After=network.target

[Service]
# User and group
User=gong
Group=gong

# Working directory
WorkingDirectory=/home/gong/gitea

# Start command
ExecStart=/home/gong/gitea/gitea web --config /home/gong/gitea/custom/conf/app.ini

# Restart policy
Restart=always
RestartSec=10s

# Environment (optional)
Environment=USER=gong HOME=/home/gong GITEA_WORK_DIR=/home/gong/gitea

# Security
PrivateTmp=true
ProtectSystem=full
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Notes:

- `WorkingDirectory` is the Gitea directory
- `ExecStart` defines the launch command
- `Restart=always` ensures auto-restart

---

### 3) Load and enable

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable gitea

# Start service
sudo systemctl start gitea

# Check status
sudo systemctl status gitea
```

Expected:

```
Active: active (running)
```

---

### 4) View logs

Real-time logs:

```
sudo journalctl -u gitea -f
```

History logs:

```
sudo journalctl -u gitea --since "1 hour ago"
```

---

## Explanation

Gitea is a Go-based self-hosted Git service. It manages local Git repos (e.g., `~/gitea/repos`) and exposes GitHub-like operations via HTTP/SSH.

Compared to `git init --bare` (just storage), Gitea adds web UI, users, PRs, and Wiki.

---

## Common issues

| Issue | Cause | Fix |
| --- | --- | --- |
| Port 3000 in use | Another service uses it | Run `./gitea web --port 8080` |
| Permission errors | Gitea runs as current user | Check repo directory permissions |
| Push fails | Repo init conflict | Do not select "Initialize with README" |
| Push slow/timeouts | Using HTTP not SSH | Configure SSH keys for faster pushes |

---

## Best practices

- SQLite is enough for personal or small teams
- Run in background: `nohup ./gitea web &`
- Backup regularly:

  ```
  ~/gitea/repos/
  ~/gitea/data/gitea.db
  ~/gitea/custom/conf/app.ini
  ```

- If the team grows, move to a server or Docker

---

## Summary

You have:

1. Deployed Gitea locally
2. Avoided port conflicts
3. Pushed existing repos to Gitea
4. Gained a web UI, PRs, and history

You now have your own private "GitHub".

---

## References

- [Gitea docs](https://docs.gitea.io/)
- [Gitea downloads](https://dl.gitea.io/gitea/)
- [Pro Git book](https://git-scm.com/book/)
- [Forgejo](https://forgejo.org/)

---

## Meta

- Reading time: 8 minutes
- Tags: `Git`, `Gitea`, `self-hosted`, `DevOps`, `version-control`
- SEO keywords: `local Gitea install`, `self-hosted Git server`, `private GitHub`, `import local repo`
- Meta description: Set up a lightweight local Git server with Gitea, including PRs, web UI, and repo management.

---

## Call to Action (CTA)

Try it now:

1. Run the install commands
2. Visit http://localhost:3000
3. Create your first repo
4. Push a project

If you want automation and backup scripts, leave a comment and I will share a follow-up.
