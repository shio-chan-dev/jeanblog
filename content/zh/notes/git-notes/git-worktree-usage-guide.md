---
title: "Git Worktree 使用教程：同仓库并行开发多个分支"
date: 2026-02-25T09:27:18+08:00
draft: false
tags: ["Git", "git worktree", "分支管理", "hotfix", "并行开发"]
description: "一篇讲清 Git worktree 常用命令、典型场景、常见坑与最佳实践的中文教程，适合多分支并行开发与临时 hotfix。"
keywords: ["git worktree", "git worktree add", "git worktree remove", "Git 并行开发", "hotfix 分支"]
---

### **标题**

Git Worktree 使用教程：同仓库并行开发多个分支

---

### **副标题 / 摘要**

`git worktree` 让你在同一个 Git 仓库下，同时打开多个分支对应的工作目录，不用来回 `checkout` 和 `stash`。  
本文覆盖常用命令、典型场景、常见坑，以及你最关心的 hotfix 分支创建方式（先进入 worktree 再建分支 / 一步建分支）。

---

### **目标读者**

- 正在同时处理多个需求分支（feature / bugfix / hotfix）的开发者
- 经常需要切到 `main` 修紧急问题，但不想污染当前工作目录的人
- 需要对比不同分支代码、并行运行不同版本服务的工程师

---

### **背景 / 动机**

很多人第一次遇到并行任务时，会用这套流程：

```bash
git stash
git checkout main
# 修 bug
git checkout feature/xxx
git stash pop
```

这套流程不是不能用，但有几个问题：

- `stash` 容易忘记清理或弹错时机
- 来回切分支容易打断当前开发节奏
- 长时间运行测试/构建会占住当前工作目录
- 对比两个分支代码时不够直观

`git worktree` 的价值就是：**在一个仓库里同时拥有多个工作目录，各做各的事，互不干扰。**

---

### **核心概念**

- **worktree（工作树）**：同一仓库的一个额外工作目录
- **共享对象库**：多个 worktree 共享同一套 Git 对象数据（不是多个独立 clone）
- **独立工作目录**：每个 worktree 有自己的文件内容、当前分支、未提交修改
- **分支占用限制**：同一分支不能同时被多个 worktree 检出（Git 会保护你）
- **stash 是仓库级别**：不是 worktree 级别，多个 worktree 共享同一个 stash 列表

结构示意（ASCII）：

```text
repo/               <- 主工作区（例如 feature/payment）
repo-featureA/      <- worktree A（例如 feature/search）
repo-hotfix/        <- worktree B（例如 hotfix/login-bug）
```

---

## 一、基本原理（它到底做了什么）

正常仓库通常只有一个工作目录：

```text
repo/
  .git/
  src/
```

使用 `git worktree` 后，你可以在同级目录拉出多个平行工作目录：

```text
repo/               <- 主工作区
repo-featureA/      <- 额外 worktree
repo-hotfix/        <- 额外 worktree
```

它们的特点：

1. **共享同一个 Git 仓库对象库**（节省磁盘）
2. **各自工作目录独立**（互不影响）
3. **每个 worktree 对应一个当前分支或 detached HEAD**

这就是它和 `git clone` 最大的差别：`worktree` 更轻、更适合“同仓库多分支并行开发”。

---

## 二、常用命令（实践指南 / 步骤）

### 1）新增一个 worktree（创建新分支并检出）

推荐写法（选项放前面，更清晰）：

```bash
git worktree add -b featureA ../repo-featureA
```

说明：

- `../repo-featureA`：新工作目录路径
- `-b featureA`：创建并检出新分支 `featureA`
- 未指定起点时，默认从当前 `HEAD` 派生（取决于你在什么分支上执行）

如果你想明确从 `main` 派生，写成：

```bash
git worktree add -b featureA ../repo-featureA main
```

### 2）新增一个 worktree（检出现有分支）

```bash
git worktree add ../repo-dev dev
```

前提：`dev` 分支当前没有被其他 worktree 占用。

### 3）查看当前所有 worktree

```bash
git worktree list
```

输出示例：

```text
/path/repo            abc123 [main]
/path/repo-featureA   def456 [featureA]
```

含义：路径 + 当前提交 + 当前分支。

### 4）删除 worktree

推荐方式：

```bash
git worktree remove ../repo-featureA
```

如果目录里有未提交修改，Git 会拒绝删除。确实要删时可强制：

```bash
git worktree remove --force ../repo-featureA
```

警告：`--force` 可能丢失该 worktree 中未提交修改。

### 5）删除分支（通常在删除 worktree 之后）

```bash
git branch -d featureA
```

如果分支未合并又要强删（谨慎）：

```bash
git branch -D featureA
```

---

## 三、可运行示例（最小命令链路）

下面是一套从新建功能 worktree 到清理的完整流程：

```bash
# 在主仓库目录（例如 repo/）执行
cd /path/to/repo

# 基于 main 创建并检出新分支 feature/search 到新目录
git worktree add -b feature/search ../repo-feature-search main

# 进入新工作目录开发
cd ../repo-feature-search
git status
git branch --show-current

# 开发提交
git add .
git commit -m "feat(search): add keyword filter"

# 回到主仓库查看所有 worktree
cd ../repo
git worktree list

# 功能结束后清理（确保 worktree 内无未提交修改）
git worktree remove ../repo-feature-search
git branch -d feature/search
```

---

## 四、典型使用场景（为什么它比 stash 流程更稳）

### 场景 1：开发新功能时临时修线上 bug（最常见）

你当前在 `feature-payment` 分支开发，突然需要修 `main` 上的紧急 bug。

#### 传统方式（容易出错）

```bash
git stash
git checkout main
# 修 bug
git checkout feature-payment
git stash pop
```

风险点：

- `stash pop` 可能冲突
- 忘记 `stash` 内容含义，回收成本高
- 思路被频繁切换打断

#### worktree 方式（推荐）

```bash
# 在主仓库目录执行（当前可在 feature-payment）
git worktree add -b hotfix/example_bug ../repo-hotfix main
cd ../repo-hotfix
```

现在你在 `repo-hotfix/` 里修 bug，原目录 `repo/` 仍保持 `feature-payment` 开发状态，互不影响。

### 场景 2：同时运行不同版本服务（v1 / v2）

例如你要对比接口行为或做兼容性验证：

- `repo-v1/` 跑稳定版本
- `repo-v2/` 跑重构版本

这样你可以并行启动服务，不需要在同一目录反复切分支、重新构建。

### 场景 3：长时间测试不阻塞当前工作目录

例如在一个 worktree 里跑：

- 集成测试
- 压测
- 大型构建

同时你在另一个 worktree 继续写代码，不会被“目录占用 + 分支切换”打断。

---

## 五、进阶用法（含你问的 hotfix 分支问题）

### 1）临时检出某个 commit（detached HEAD）

```bash
git worktree add ../repo-test <commit-id>
```

这会进入 detached HEAD 状态，适合：

- 快速复现历史版本问题
- 对比某个提交点行为
- 临时验证，不打算长期保留分支

### 2）修复 “worktree 记录还在，但目录不在了”

如果你手动删除了 worktree 目录，Git 元数据里可能还留着记录，后续会报错。

清理方式：

```bash
git worktree prune
```

### 3）同一分支不能被多个 worktree 同时使用

如果报错：

```text
fatal: 'featureA' is already checked out at ...
```

说明 `featureA` 已被某个 worktree 检出。可选方案：

- 删除旧 worktree
- 切走旧 worktree 的分支
- 新建不同分支（更常见）

### 4）你问的关键问题：先进入 worktree 再 `checkout -b` 可以吗？

你给的流程：

```bash
git worktree add ../repo-hotfix main
cd ../repo-hotfix
git checkout -b hotfix/example_bug
```

结论：**可以，完全合法，而且是常见用法。**

发生了什么：

1. `git worktree add ../repo-hotfix main`
   - 新建一个 worktree 目录 `../repo-hotfix`
   - 在该 worktree 中检出 `main`
2. `git checkout -b hotfix/example_bug`
   - 基于当前 `main` 创建新分支 `hotfix/example_bug`
   - 并切换到这个新分支

执行后结构会变成：

```text
repo/         -> 例如仍在 feature-x（不受影响）
repo-hotfix/  -> hotfix/example_bug
```

#### 更推荐的一步写法（少一次 checkout）

```bash
git worktree add -b hotfix/example_bug ../repo-hotfix main
```

含义：

- 基于 `main`
- 创建 `hotfix/example_bug`
- 直接在新 worktree 检出该分支

这也是实际团队里更常用的写法。

#### 重要细节（很多人会踩坑）

如果你的主工作目录当前就在 `main`，那么下面这条命令可能会失败：

```bash
git worktree add ../repo-hotfix main
```

因为同一个分支（`main`）不能同时被两个 worktree 检出。

而这条一步写法通常可以工作：

```bash
git worktree add -b hotfix/example_bug ../repo-hotfix main
```

原因是新 worktree 最终检出的是 `hotfix/example_bug`，`main` 只是作为起点（start-point），不是最终被占用的分支。

#### 关于 “main 会被释放” 的准确说法

当你在 `repo-hotfix` 执行：

```bash
git checkout -b hotfix/example_bug
```

该 worktree 会从 `main` 切到 `hotfix/example_bug`，因此：

- **这个 worktree 不再占用 `main`**
- 但 `main` 是否被其他 worktree 占用，取决于你仓库的整体状态（例如主目录是否仍在 `main`）

---

## 六、解释与原理（为什么 worktree 好用）

`git worktree` 的核心不是“多目录”本身，而是它把两类状态拆开了：

- **仓库历史对象（共享）**
- **工作目录状态（独立）**

这样你就可以：

- 共享 Git 历史与对象库（节省磁盘）
- 独立维护每个任务的工作现场（减少切换风险）

相比 `clone`：

- `clone` 更独立，但更重
- `worktree` 更轻，适合同仓库多分支并行开发

---

## 七、常见问题与注意事项（FAQ + 坑点）

### 1）一个分支只能在一个 worktree 使用吗？

是的。Git 会保护你，避免同一分支在多个工作目录同时修改造成混乱。

### 2）`stash` 是不是 worktree 独立的？

不是。`stash` 是仓库级共享的。  
所以如果你在多个 worktree 里频繁使用 `stash`，要特别注意命名和使用顺序。

### 3）submodule 怎么办？

`worktree` 不会自动把 submodule 状态处理到你预期一致。进入新 worktree 后，通常需要单独执行：

```bash
git submodule update --init --recursive
```

### 4）手动删掉 worktree 目录后报错怎么办？

执行：

```bash
git worktree prune
```

清理掉 Git 元数据里的残留记录。

### 5）可以嵌套 worktree 吗？

不建议。路径关系会非常混乱，排查问题成本高。

### 6）`git worktree remove --force` 什么时候用？

仅在你确认不需要保留该 worktree 未提交修改时使用。否则可能直接丢失改动。

---

## 八、和 clone 的区别（对比表）

| 对比项 | `git worktree` | `git clone` |
| --- | --- | --- |
| 是否共享 `.git` 对象库 | ✅ 是 | ❌ 否 |
| 磁盘占用 | 少 | 多 |
| 是否独立仓库 | 否（共享仓库历史） | 是 |
| 适合多分支并行开发 | 非常适合 | 一般 |
| 适合完全隔离实验环境 | 一般 | 更适合 |

结论：

- **同一仓库多分支并行开发**：优先 `worktree`
- **完全隔离仓库配置/远程/钩子实验**：考虑 `clone`

---

## 九、推荐实践（目录与流程）

### 推荐目录结构

```text
project-main/
project-feature1/
project-feature2/
project-hotfix/
```

或者保留主目录名不变：

```text
project/                <- 主工作区
project-feature1/
project-feature2/
project-hotfix/
```

### 常用快捷命令（建议收藏）

```bash
# 新建 feature（基于 main）
git worktree add -b feature/xxx ../proj-feature-xxx main

# 查看所有 worktree
git worktree list

# 删除 worktree
git worktree remove ../proj-feature-xxx

# 清理手动删除后的残留记录
git worktree prune
```

### 热修复推荐流程（实战版）

```bash
# 当前目录可能在 feature 分支中开发
cd /path/to/repo

# 基于 main 拉出 hotfix worktree，并直接创建 hotfix 分支
git worktree add -b hotfix/example_bug ../repo-hotfix main
cd ../repo-hotfix

# 修复并提交
git add .
git commit -m "fix: patch example bug"
git push origin hotfix/example_bug

# 清理（确认已不需要该目录）
cd ../repo
git worktree remove ../repo-hotfix
git branch -d hotfix/example_bug
```

说明：最后一条 `git branch -d` 只有在你本地还保留该分支引用且已合并时才会成功；未合并请先确认再处理。

---

## 十、小结 / 结论

一句话总结：

**`git worktree` = 在一个 Git 仓库里开多个“平行宇宙工作目录”。**

你最该记住的三件事：

1. 并行开发时，优先用 `worktree` 替代频繁 `stash + checkout`
2. hotfix 场景更推荐一步式写法：`git worktree add -b hotfix/... ../path main`
3. 同一分支不能被多个 worktree 同时检出，这是 Git 的保护机制，不是 bug

---

## 十一、参考与延伸阅读

- Git 官方文档：`git worktree`（建议直接看 `git help worktree` 或官方 manpage）
- Git Book（官方 Pro Git）中关于多工作目录/分支管理的相关章节
- 你仓库内相关文章：`content/zh/notes/git-notes/git-branching-workflow.md`

---

## 十二、元信息

- **阅读时长**：约 8-10 分钟
- **标签**：Git、git worktree、分支管理、hotfix、并行开发
- **SEO 关键词**：git worktree, git worktree add, git worktree remove, Git 多分支并行开发
- **元描述**：一篇讲清 Git worktree 常用用法与常见坑的中文教程，覆盖 hotfix 场景、分支占用限制、prune 清理与 clone 对比。

---

## 十三、行动号召（CTA）

如果你愿意，我下一篇可以继续写这两个进阶主题里的任意一个：

1. `git worktree + rebase` 的常见坑（尤其是多个 worktree 同时改同一功能链）
2. 一个真实项目的 `worktree` 最佳实践示例（含目录命名、脚本化创建与清理）
