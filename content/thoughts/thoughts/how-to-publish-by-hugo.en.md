---
title: "How to Publish with Hugo"
date: 2025-11-14T15:01:32+08:00
---

# How to Publish with Hugo: From Markdown to Online Blog

## Subtitle / Abstract

This guide explains how to create, manage, and publish Hugo posts: front matter, drafts, images, directory structure, local preview, and deployment.

---

## Target readers

- Hugo beginners
- Developers building a technical blog with Hugo
- Writers using Markdown + static sites
- Users of PaperMod, DoIt, and similar themes

---

## Background / Motivation

After setting up a Hugo site, common questions include:

- Where should posts go?
- How should front matter be written?
- Where do images live?
- Why does the post show locally but not online?
- How do drafts and publish dates work?
- How do posts appear on the homepage?

This guide provides practical steps and best practices for the full publishing flow.

---

## Core concepts

### 1) Content directory

Hugo posts live under `content/`:

```
content/
  posts/
    my-first-post.md
```

### 2) Front matter

Top metadata controls title, date, draft, tags, etc.

```yaml
---
title: "My Title"
date: 2024-08-26
draft: false
tags: ["hugo", "blog"]
---
```

### 3) Draft

Drafts are not built. Use `hugo server -D` to preview drafts.

### 4) Section

`content/posts/*` maps to the `/posts/` section.

---

## Steps

### Step 1: Create a new post

```bash
hugo new posts/how-to-publish.md
```

This creates:

```
content/posts/how-to-publish.md
```

Default content:

```yaml
---
title: "How to Publish"
date: 2024-08-26T10:00:00+08:00
draft: true
---
```

---

### Step 2: Edit front matter

A typical PaperMod-friendly front matter:

```yaml
---
title: "How to Publish with Hugo"
date: 2024-08-26T10:00:00+08:00
draft: false
tags: ["hugo", "blog", "static-site"]
categories: ["tutorial"]
summary: "A complete guide from writing to publishing."
cover:
  image: "/images/hugo-cover.png"
  alt: "Hugo cover"
  caption: "Hugo blog cover"
---
```

---

### Step 3: Write content

Use Markdown and keep headings consistent. Add code blocks where needed.

---

### Step 4: Add images

Common options:

- `static/images/...` and reference as `/images/...`
- Page bundles: `content/posts/my-post/index.md` with images in the same folder

---

### Step 5: Preview locally

```bash
hugo server -D
```

Open `http://localhost:1313`.

---

### Step 6: Build and deploy

```bash
hugo
```

Static output goes to `public/`. Deploy via GitHub Pages, Netlify, or your server.

---

## Common pitfalls

- `draft: true` prevents publishing
- Wrong folder (e.g., outside `content/`)
- Missing or incorrect `baseURL`
- Images referenced with wrong paths
- Theme config not loaded

---

## Summary

- Create posts with `hugo new`
- Write front matter carefully
- Preview with `hugo server -D`
- Build with `hugo` and deploy

If you want a complete deployment pipeline with GitHub Actions, see the next guide.
