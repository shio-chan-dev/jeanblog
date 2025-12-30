---
title: "How to Create and Edit Mermaid Diagrams"
date: 2025-08-26
draft: false
---

# Introduction

Mermaid is a framework for creating diagrams using code. This post shows how to install the tooling on your server and render Mermaid code into images.

# Steps

## Install the renderer

Run:

```bash
npm install -g @mermaid-js/mermaid-cli
```

Note: the CLI requires npm version >= 20. It is recommended to manage npm versions with nvm.

If you do not have nvm, install it with:

```bash
curl -o https://raw.githubusercontent.com/nvm-sh/nvim/v0.39.4/install.sh | bash
```

Restart your shell, then run:

```bash
nvm install 20
nvm use 20
nvm alias default 20
```

Verify:

```
node -v
npm -v
```

## Render a diagram

Put your Mermaid code in a file ending with `.mmd`, then run:

```bash
mmdc -i diagrams/example.mmd -o images/example.svg
```
