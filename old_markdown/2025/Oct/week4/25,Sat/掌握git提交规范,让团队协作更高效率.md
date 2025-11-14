### **标题**

🚀 从「feat」到「fix」：掌握 Git 提交规范，让团队协作与自动化更高效

---

### **副标题 / 摘要**

一篇为开发者准备的实用指南，带你理解并掌握业界通行的 Git 提交信息标准（Conventional Commits），
从 commit 标签（如 `feat:`、`fix:`）到自动生成 changelog，一次学会写出高质量的提交记录。

---

### **目标读者**

* **初学者**：刚开始使用 Git，想养成规范提交的习惯。
* **中级开发者**：希望让提交信息对团队和 CI 工具更友好。
* **团队负责人 / 架构师**：想建立统一的代码提交标准，提升协作与版本管理效率。

---

### **背景 / 动机**

大多数开发者写提交信息的方式都是这样的：

> “update code”
> “fix bug”
> “修改东西”

这类信息短期可读，长期无用。
当团队人数增多、项目复杂时，**无法追踪改动意图**，也无法让自动化工具正确识别变更类型。
这就是为什么业界推出了 **Conventional Commits**：
一个简洁统一的 commit 语法标准，让 Git 提交**可读、可追踪、可自动化**。

---

### **核心概念**

**Conventional Commits** 是一种提交信息格式约定，它规定了提交消息的结构：

```
<type>(<scope>): <subject>

<body>

<footer>
```

* `type`：提交类型，如 `feat`、`fix`、`docs`
* `scope`：作用范围，可选（如 `ui`、`api`）
* `subject`：简短描述（不超过 50 字）
* `body`：详细说明（可选）
* `footer`：备注（如 BREAKING CHANGE）

---

### **实践指南 / 步骤**

1️⃣ **设置 Git 编辑器为 Neovim（可选）**

```bash
git config --global core.editor "nvim"
```

2️⃣ **编写标准化的提交信息**

```bash
git commit -m "feat(lsp): 适配新版 nvim-lspconfig 接口"
```

3️⃣ **提交规范结构示例**

```
feat(lsp): 更新 LSP 配置以适配新版 nvim-lspconfig

- 删除旧写法 lspconfig[server].setup
- 改用新函数调用形式 lspconfig(server, {...})
```

4️⃣ **使用工具强制检查规范（可选）**

```bash
npm install -g commitlint @commitlint/config-conventional
```

添加配置文件 `.commitlintrc.js`：

```js
module.exports = { extends: ["@commitlint/config-conventional"] };
```

---

### **可运行示例**

```bash
# 新功能提交
git commit -m "feat(auth): 支持双因素登录"

# 修复 bug
git commit -m "fix(ui): 修复暗色模式下文字不可见"

# 更新文档
git commit -m "docs(readme): 补充使用说明"

# 重构
git commit -m "refactor(api): 优化用户认证逻辑"

# 性能优化
git commit -m "perf(db): 提升查询缓存效率"
```

---

### **解释与原理**

这套规范来自 **Angular 团队的 commit message 格式**，
后被广泛采纳为开源标准（Conventional Commits）。

**优势：**

* 结构清晰：一眼看出类型与范围
* 机器可读：可自动生成 changelog
* 易于集成：配合 `semantic-release` 自动生成版本号

**替代方案：**

* [Gitmoji](https://gitmoji.dev/)（使用 emoji 提交）
* [Semantic Versioning](https://semver.org/)（配合自动发版）

---

### **常见问题与注意事项**

| 问题           | 说明                    |
| ------------ | --------------------- |
| 我能混用中文和英文吗？  | 可以，推荐标题英文、内容中文，保持一致性。 |
| 一次提交多个类型怎么办？ | 拆分为多次提交，每次只做一类事。      |
| 提交太短没内容怎么办？  | 至少写清楚「为什么改」。          |
| 是否必须写 scope？ | 可选，但推荐加上模块名或功能域。      |

---

### **最佳实践与建议**

✅ 保持一条提交只做“一件事”
✅ 标题首字母小写，不加句号
✅ 第一行 ≤ 50 字
✅ 第二行空一行，第三行开始写详细描述
✅ 使用动词开头（如 add、fix、update）

---

### **小结 / 结论**

规范化 commit 信息是一种**小投入、大回报**的习惯。
它让项目更可维护、让团队沟通更顺畅、让自动化工具帮你节省时间。
写好 commit message = 写给未来的自己和队友看的 changelog。

---

### **参考与延伸阅读**

* 📘 [Conventional Commits 官方规范](https://www.conventionalcommits.org/)
* 📗 [Angular Commit Message Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
* 🧩 [semantic-release](https://semantic-release.gitbook.io/)
* 🧠 [Gitmoji](https://gitmoji.dev/)

---

### **元信息**

* **阅读时长**：约 6 分钟
* **标签**：Git、开发规范、Conventional Commits、团队协作
* **SEO 关键词**：Git 提交规范、Conventional Commits、feat fix refactor、提交信息最佳实践
* **元描述**：一篇为开发者准备的 Git 提交规范指南，教你用 `feat:`、`fix:` 等标准化格式写出清晰、可维护的 commit message。

---

### **行动号召（CTA）**

💪 尝试为你下一个提交加上正确的标签吧：

```bash
git commit -m "feat: 初次使用提交规范 🚀"
```

👉 或者在评论区分享你团队的提交风格，让更多人少走弯路。

