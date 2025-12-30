---
title: "How to Build a Blog System"
date: 2025-11-14T15:04:02+08:00
---

# Build a Hugo Blog with GitHub Pages in 10 Minutes

## Subtitle / Abstract

This guide takes you from zero to a deployed Hugo blog on GitHub Pages with GitHub Actions. It is beginner-friendly and explains the key moving parts.

---

## Target readers

- Hugo beginners
- Developers who want a quick technical blog
- Users of GitHub Pages and GitHub Actions
- Anyone who wants free static hosting

---

## Background / Motivation

Common pain points when publishing a blog:

- manual uploads
- scattered deployment steps
- confusing GitHub Pages setup
- theme assets failing to build

The combo **Hugo + GitHub Pages + GitHub Actions** solves these:

- Hugo is fast
- Pages is free
- Actions deploys on every push

---

## Core concepts

- **Hugo**: static site generator
- **GitHub Pages**: free static hosting
- **GitHub Actions**: CI pipeline to build and deploy
- **PaperMod**: popular Hugo theme

---

## Steps: from local to online

### Step 1: Create a Hugo site

```bash
hugo new site myblog
cd myblog
git init
```

Add PaperMod:

```bash
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

Set `config.toml`:

```toml
baseURL = "https://<your-username>.github.io/<repo>/"
languageCode = "en-us"
title = "My Blog"
theme = "PaperMod"
```

---

### Step 2: Push to GitHub

```bash
git remote add origin git@github.com:<your-username>/<repo>.git
git add .
git commit -m "init blog"
git push -u origin main
```

---

### Step 3: Enable GitHub Pages

In your repo:

- Settings -> Pages
- Build and deployment -> Source = GitHub Actions

If the repo is private, set it public or enable Pages for private repositories (paid). Otherwise you may see 404.

---

### Step 4: Add GitHub Actions workflow

Create `.github/workflows/hugo.yml`:

```yaml
name: Deploy Hugo site to Pages

on:
  push:
    branches: ["main"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true
      - uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: "0.120.4"
      - name: Build
        run: hugo --minify
      - name: Upload
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./public

  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

---

### Step 5: Create a post and publish

```bash
hugo new posts/hello-world.md
```

Edit front matter, set `draft: false`, write your content, and push.

---

## Common issues

- `draft: true` means the post will not show
- incorrect `baseURL` causes broken links
- missing submodules in Actions -> theme missing
- Pages source not set to Actions

---

## Summary

- Hugo builds fast static pages
- GitHub Actions automates build and deploy
- GitHub Pages hosts for free

Once set, you only need to write Markdown and push.
