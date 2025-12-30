---
title: "Use a Local Git Bare Repo to Separate Dev and Test Environments"
date: 2025-10-20
draft: false
---

# Use a Local Git Bare Repo to Separate Dev and Test Environments

In full-stack work, a common problem is **how to isolate dev and test environments**. Many people host on GitHub or GitLab, but private projects may not be suitable for public hosting.

Git is distributed. You can set up a **local bare repo** as a remote to move code from **dev -> test** in one machine.

---

## What is a bare repository?

- A normal repo (`git init`) has a **working tree + .git metadata** and can be edited directly.
- A bare repo (`git init --bare`) has only Git data, no working tree. It is usually used as a **remote**.

In short:

- **Dev repo**: where you write code
- **Bare repo**: remote sync point with full history
- **Test repo**: clone from bare repo to simulate deployment

---

## Step 1: Create the bare repo

Create a bare repo under a local directory (e.g., `~/.repos`):

```bash
mkdir -p ~/.repos
cd ~/.repos
git init --bare scrapy.git
```

Now `~/.repos/scrapy.git` is your local remote.

---

## Step 2: Add the local remote in your dev repo

Assume your dev repo is `~/scrapy`:

```bash
cd ~/scrapy
git remote add local ~/.repos/scrapy.git
```

Check:

```bash
git remote -v
```

Expected:

```
local   /home/gong/.repos/scrapy.git (fetch)
local   /home/gong/.repos/scrapy.git (push)
```

---

## Step 3: Push to the local remote

Push `main`:

```bash
git push local main
```

Your bare repo now contains all commits.

---

## Step 4: Clone in the test environment

Assume your test environment is `~/test-env`:

```bash
cd ~/test-env
git clone ~/.repos/scrapy.git
```

You now have a clean copy for testing without affecting dev.

---

### Note on HEAD warning

Sometimes you see:

```pgsql
warning: remote HEAD refers to nonexistent ref, unable to checkout
```

This happens because a newly created bare repo has no default HEAD. Set it:

```bash
cd ~/.repos/scrapy.git
git symbolic-ref HEAD refs/heads/main
```

Then clone again.

---

## Step 5: Sync workflow

- In dev (`~/scrapy`):

  ```bash
  git add .
  git commit -m "feat: finish feature"
  git push local main
  ```

- In test (`~/test-env/scrapy`):

  ```bash
  git pull
  ```

Now you can sync **dev -> test** easily on one machine.

---

## Summary

If you cannot push to GitHub/GitLab, a local bare repo can separate dev and test:

- no external platform required
- dev and test isolated
- full Git history preserved

If the project grows, consider a private Git service (Gitea/GitLab CE) or Docker deployment.
