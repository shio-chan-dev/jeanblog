---
title: "How to Build a Blog System"
date: 2025-11-14T15:04:02+08:00
---
# **标题：用 Hugo + GitHub Pages 十分钟上线个人博客（超详细新手指南）**

## **副标题 / 摘要**

本教程带你从零开始，将本地 Hugo 博客部署到 GitHub Pages，全程只需 10 分钟，适合想快速上线技术博客、文档站点的开发者。确保你不仅能跑起来，还能理解背后的工作原理。

---

## **目标读者**

* Hugo 初学者
* 想快速上线个人技术博客的开发者
* 想了解 GitHub Pages + GitHub Actions 部署的用户
* 想要零成本托管静态网站的同学

---

## **背景 / 动机：为什么要用 Hugo + GitHub Pages？**

许多人写博客时面临这些痛点：

* 发布文章要手动上传，不自动化
* 静态站点生成器很多，但部署步骤零散
* GitHub Pages 文档不够清晰，新手容易踩坑
* 主题（如 PaperMod）需要正确处理资源（SCSS）才能编译成功

**Hugo + GitHub Pages + GitHub Actions 组合** 完美解决了这些问题：

* Hugo 构建速度极快（上千文章依旧瞬间生成）
* GitHub Pages 完全免费，不需要服务器
* GitHub Actions 自动部署，写完文章 push 即上线

---

## **核心概念（必须理解）**

### **1. Hugo**

一个超快的静态博客生成器，通过 Markdown 生成 HTML。

### **2. GitHub Pages**

GitHub 提供的免费静态网站托管。

### **3. GitHub Actions**

GitHub 的自动化流水线，用来：

* 安装 Hugo
* 构建你的博客
* 部署到 Pages

### **4. PaperMod**

Hugo 最流行的主题之一，外观现代、适合技术博客。

---

## **实践指南 / 步骤：从零到上线**

以下是完整步骤，你只需要照做即可。

---

### **✔ 第 1 步：设置 Hugo 博客项目**

（你已完成）

```bash
hugo new site myblog
cd myblog
git init
```

安装 PaperMod：

```bash
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

修改 `config.toml`：

```toml
baseURL = "https://shio-chan-dev.github.io/jeanblog/"
languageCode = "zh-cn"
title = "Jean’s Blog"
theme = "PaperMod"
```

---

### **✔ 第 2 步：推送到 GitHub 仓库**

```bash
git remote add origin git@github.com:shio-chan-dev/jeanblog.git
git add .
git commit -m "init blog"
git push -u origin main
```

---

### **✔ 第 3 步：启用 GitHub Pages（关键）**

进入你的仓库：
`https://github.com/shio-chan-dev/jeanblog`

点击：

* Settings
* Pages
* Build and deployment
* **Source = GitHub Actions**（必须、关键）

如果仓库是 Private，请改成 Public（否则 Pages 返回 404）。

---

### **✔ 第 4 步：添加 GitHub Actions 工作流**

创建文件：

```
.github/workflows/hugo.yml
```

内容如下：

```yaml
name: Deploy Hugo site to GitHub Pages

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: "latest"
          extended: true

      - name: Build
        run: hugo --minify

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

提交：

```bash
git add .
git commit -m "add github pages workflow"
git push
```

---

### **✔ 第 5 步：等待部署完成**

进入仓库 → Actions
等待 `Deploy Hugo site to GitHub Pages` 变成绿色 ✔

成功后，你会在：

**Settings → Pages**

看到提示：

📢 *Your site is live at:*
[https://shio-chan-dev.github.io/jeanblog/](https://shio-chan-dev.github.io/jeanblog/)

现在博客上线了 🎉

---

## **可运行示例（复制即用）：单篇文章 front matter**

以下 front matter 适用于 PaperMod：

```yaml
---
title: "如何部署 Hugo 博客到 GitHub Pages"
date: 2024-08-26T10:00:00+08:00
draft: false
tags: ["hugo", "github pages"]
summary: "最清晰的 Hugo + GitHub Pages 部署教程。"
---
```

---

## **解释与原理（为什么这么做？）**

### **1. 为什么必须用 GitHub Actions 部署？**

因为 PaperMod 用了 SCSS，需要 Hugo Extended 才能编译。

GitHub Pages 内置的 Jekyll 无法处理 Hugo 构建 → 必须用 Actions。

---

### **2. 为什么 baseURL 不能写错？**

因为静态文件路径依赖 baseURL。
GitHub Pages 的 Project Pages 必须加仓库名：

```
https://用户名.github.io/仓库名/
```

写成 `/` 或根目录会导致 CSS/JS 加载失败。

---

### **3. 为什么 Private 仓库会返回 404？**

GitHub 免费用户仅允许 Public 仓库使用 Pages。
Private 仓库部署会直接报：

```
Failed to create deployment (404)
```

---

## **常见问题与注意事项**

### ❌ 本地能跑，GitHub Pages 404？

→ 你没开启 Pages（Settings → Pages）
→ `baseURL` 写错
→ 仓库是 Private
→ Actions 构建失败（去 Actions 看 log）

---

### ❌ 部署成功但样式丢失？

→ 99% 是 `baseURL` 错了
→ PaperMod 主题路径加载不到

---

### ❌ 文章本地能显示，但线上不见？

→ `draft: true`
→ 没有 push 到 main
→ build 失败

---

## **最佳实践与建议**

* **仓库务必使用 Public**（免费 Pages 功能）
* **把 config.toml 放入版本控制**
* **使用 GitHub Actions 自动部署，不要手动上传 public 文件夹**
* 写文章时使用日期排序，不用刻意修改文件名
* 为博客开启：`showReadingTime`, `showToc`, `showBreadCrumbs`

---

## **小结 / 结论**

你已经完成：

* 搭建 Hugo 博客
* 使用 PaperMod 主题
* 配置 GitHub Actions 自动部署
* 成功上线 GitHub Pages 网站

从此以后，你的博客更新非常简单：

```bash
hugo server -D  # 本地预览
git add .
git commit -m "new post"
git push        # 自动上线
```

这是最省心、最现代化的写作方式之一。

---

## **参考与延伸阅读**

* Hugo 官方文档
  [https://gohugo.io/](https://gohugo.io/)
* PaperMod 主题
  [https://github.com/adityatelange/hugo-PaperMod](https://github.com/adityatelange/hugo-PaperMod)
* GitHub Pages
  [https://pages.github.com/](https://pages.github.com/)
* GitHub Actions 文档
  [https://docs.github.com/en/actions](https://docs.github.com/en/actions)

---

## **文章元信息**

* **阅读时间：8–12 分钟**
* **标签：Hugo、GitHub Pages、PaperMod、静态博客、自动部署**
* **SEO 关键词：Hugo 部署、GitHub Pages 教程、PaperMod、静态网站、博客搭建**
* **元描述：用 Hugo + GitHub Pages 搭建个人博客的完整指南，从初始化到上线，适用于任何级别的开发者。**

---

## **行动号召（CTA）**

如果你已经成功部署自己的 Hugo 博客，不妨：

* ⭐ 收藏文章，以便未来重做环境时快速参考
* 💬 在评论区分享你的博客地址
* 🔄 尝试部署到 Vercel / Cloudflare Pages（提升访问速度）
* 📝 开始写你的第一篇文章，记录构建博客的过程



