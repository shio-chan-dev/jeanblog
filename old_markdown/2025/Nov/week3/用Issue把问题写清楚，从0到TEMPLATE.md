## 标题（吸引且准确，包含关键词）

**用 Issue 模板把需求写清楚：从 0 配置 GitHub Issue Template 的完整指南**

---

## 副标题 / 摘要

这篇文章手把手教你在 GitHub 仓库中配置「新需求 / Feature」与「Bug」Issue 模板，包括目录结构、YAML 表单、Markdown 模板以及常见坑。适合想让团队需求沟通更规范、减少反复追问的开发者和团队负责人。

---

## 目标读者

这篇文章适合：

* 经常在 GitHub 仓库里开 Issue、提需求的 **后端 / 前端 / 全栈工程师**
* 想把团队需求提交流程「标准化」的 **项目负责人 / TL / 架构师**
* 对 GitHub 已经有基本使用经验、但还没用过 Issue 模板的 **中级开发者**

完全新手也能看懂，但会默认你知道：什么是仓库、什么是 Issue、如何提交代码等。

---

## 背景 / 动机：为什么要折腾 Issue 模板？

没有 Issue 模板时，日常可能是这样的：

* “这个需求背景是什么？”
* “影响哪些模块？”
* “验收标准怎么算通过？”
* “优先级到底多高？”

一句话 Issue：

> “做个导出功能”
> 直接把所有人整破防。

长期下来会有几个痛点：

1. **沟通成本高**：每个需求都要反复追问细节；
2. **信息不对称**：请求人脑子里很清楚，但写在 Issue 里的只有一句话；
3. **难以排期**：没有明确优先级和验收标准，大家都觉得自己的需求是 P0；
4. **历史难追踪**：几个月后再看这个 Issue，完全不知道当时怎么想的。

而 GitHub 提供的 **Issue Template**，其实就是一套「结构化提问」工具：

* 新建 Issue 时强制/引导用户按模板填写；
* 自动带上标签、标题前缀；
* 可以用表单形式校验必填项。

**目标**很简单：让每一个新需求一眼就能看懂，减少沟通折腾。

---

## 核心概念：我们要搞懂的几个关键词

在配置之前，先把几个概念说清楚：

### 1. Issue Template（Issue 模板）

* 新建 Issue 时出现的“预设格式”
* 可以是纯文本（Markdown），也可以是 Web 表单（YAML）

### 2. Markdown 模板

* 旧式 / 简单版
* 本质上就是一个预填的 Markdown 文本
* 文件放在：`.github/ISSUE_TEMPLATE/xxx.md` 或 `.github/ISSUE_TEMPLATE.md`

### 3. YAML Issue 表单（表单模板）

* 新式 / 推荐
* 新建 Issue 时会出现带输入框、下拉框的表单
* 提交后会把你的填写内容转成 Markdown 填进 Issue 正文
* 文件放在：`.github/ISSUE_TEMPLATE/xxx.yml`

### 4. config.yml

* 放在：`.github/ISSUE_TEMPLATE/config.yml`
* 控制：

  * 是否允许“没有模板的空白 Issue”
  * 模板列表的显示（部分场景）

---

## 实践指南 / 步骤概览

我们按下面这个顺序来做：

1. 创建 `.github/ISSUE_TEMPLATE` 目录
2. 新建「新需求 / Feature」模板（YAML 表单）
   3.（可选）新建「Bug 反馈」模板
3. 配置 `config.yml` 控制是否允许空白 Issue
4. 提交并推送到 GitHub
5. 在 Web 上验证模板是否生效

---

## 步骤一：创建 Issue 模板目录

在你的项目根目录下执行：

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

创建完后，目录结构大致是：

```text
your-repo/
  .github/
    ISSUE_TEMPLATE/
      # 等会儿我们会往这里加 yml / md 文件
  src/
  ...
```

---

## 步骤二：创建「新需求 / Feature」模板（YAML 表单）

在 `.github/ISSUE_TEMPLATE/feature-request.yml` 中写入以下内容：

```yaml
name: "新需求 / Feature"
description: "用于提交新的功能需求或需求变更"
title: "[需求] "
labels:
  - "feature"
  - "enhancement"

body:
  - type: markdown
    attributes:
      value: |
        感谢提交新需求 🙏  
        请尽量填写清晰，方便评估和排期。

  - type: input
    id: module
    attributes:
      label: 影响模块
      description: 涉及的服务/模块，例如：后端接口、爬虫、前端页面等
      placeholder: 例如：ecp 爬虫 / 附件浏览接口
    validations:
      required: true

  - type: textarea
    id: background
    attributes:
      label: 背景 / 场景
      description: 为什么要做这个需求？当前遇到什么问题？有没有现有替代方案？
      placeholder: |
        简要描述业务背景、角色、使用场景、痛点等…
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: 需求描述
      description: 希望系统具体怎么变化？最好从“用户视角”来描述。
      placeholder: |
        1. 在 xxx 页面增加 ...
        2. 当用户执行 ... 时，系统应 ...
        3. 需要支持的边界场景：...
    validations:
      required: true

  - type: textarea
    id: acceptance_criteria
    attributes:
      label: 验收标准
      description: 哪些情况算是“满足需求”？方便后续自测和验收。
      placeholder: |
        - [ ] 场景一：...
        - [ ] 场景二：...
        - [ ] 性能 / 安全性要求：...
    validations:
      required: true

  - type: dropdown
    id: priority
    attributes:
      label: 优先级
      description: 方便排期排序
      options:
        - P0（必须本迭代完成）
        - P1（高优先级）
        - P2（一般）
        - P3（低）
      default: 2
    validations:
      required: false

  - type: textarea
    id: extra
    attributes:
      label: 其他信息
      description: 相关接口、文档链接、设计稿、截图、关联 Issue 等
      placeholder: |
        - 接口文档：
        - 设计稿 / 原型：
        - 相关 Issue / 需求单：
    validations:
      required: false
```

**效果：**

* 新建 Issue 时会有一个「新需求 / Feature」选项；
* 点进去是表单，而不是纯文本；
* `labels` 会自动打上 `feature` / `enhancement` 标签；
* 标题自动带 `[需求] ` 前缀；
* `background/description/acceptance_criteria` 等字段是必填。

---

## 步骤三（可选）：创建「Bug 反馈」模板

在 `.github/ISSUE_TEMPLATE/bug-report.yml` 写入：

```yaml
name: "缺陷 / Bug"
description: "用于提交 Bug 和异常问题"
title: "[Bug] "
labels:
  - "bug"

body:
  - type: textarea
    id: summary
    attributes:
      label: 问题概述
      placeholder: 简要描述问题现象
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: 复现步骤
      placeholder: |
        1. 打开 ...
        2. 点击 ...
        3. 看到 ...
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: 预期结果
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: 实际结果
    validations:
      required: true

  - type: textarea
    id: extra
    attributes:
      label: 其他信息
      description: 日志、截图、环境信息等
    validations:
      required: false
```

这样一来，团队就可以比较清楚地区分「功能需求」和「Bug」。

---

## 步骤四：配置 `config.yml`（控制模板选择和空白 Issue）

在 `.github/ISSUE_TEMPLATE/config.yml` 写入：

```yaml
blank_issues_enabled: false  # 禁止直接新建“空白 Issue”，强制选模板
contact_links:
  - name: 内部需求管理系统
    url: https://example.com/your-internal-system
    about: 如为正式立项需求，请先在内部系统中创建，再在此关联编号。
```

如果你还没内部需求系统，可以先把 `contact_links` 删除或改成你自己的 Wiki 链接。

`blank_issues_enabled: false` 会让所有 Issue 都必须走模板，避免出现“什么都没填就扔一个 Issue”的情况。

---

## 步骤五：提交并推送到 GitHub

```bash
git add .github/ISSUE_TEMPLATE/*
git commit -m "chore: add GitHub issue templates for feature & bug"
git push
```

推送到默认分支（通常是 `main` 或 `master`）之后，模板就生效了。

---

## 步骤六：在 GitHub 上验证效果

1. 打开你的 GitHub 仓库；
2. 点击上方的 **Issues**；
3. 点击 **New issue**。

此时一般会看到一个「选择模板」的页面，例如：

* 新需求 / Feature
* 缺陷 / Bug
* （如果开了）Open a blank issue

如果你配置了 `blank_issues_enabled: false`，就不会有空白 Issue 选项。

点「新需求 / Feature」，你会看见你刚才在 YAML 里定义的表单，中英文都能正常显示。

---

## 可运行示例：最小可用配置（拷贝即用）

如果你只想要**一套最小可用**的 Feature 模板，下面这两步就够了：

**1）创建目录：**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**2）创建 `.github/ISSUE_TEMPLATE/feature-request.yml`：**

```yaml
name: "新需求 / Feature"
description: "用于提交新的功能需求或需求变更"
title: "[需求] "
labels: ["feature"]

body:
  - type: textarea
    id: background
    attributes:
      label: 背景 / 场景
      placeholder: 简要描述为什么要做这个需求
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: 需求描述
      placeholder: |
        希望系统做什么？用户如何使用？列出关键流程。
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: 验收标准
      placeholder: |
        - [ ] 场景一：...
        - [ ] 场景二：...
    validations:
      required: true
```

加上：

```bash
git add .github/ISSUE_TEMPLATE/feature-request.yml
git commit -m "add minimal feature request issue template"
git push
```

就能在仓库里看到一个「新需求 / Feature」模板。

---

## 解释与原理：为什么要用 YAML 表单而不是单纯 Markdown？

### YAML 表单的好处

* **必填校验**：可以强制要求填写“背景”“需求描述”“验收标准”等，避免空空如也；
* **更友好的 UI**：对非技术同事也相对友好，不需要懂 Markdown；
* **结构更清晰**：每个字段都是独立的，便于阅读和后期自动化处理（比如机器人、脚本）；
* **自动打标签 / 标题前缀**：省去后续人工维护。

### Markdown 模板的优势与局限

Markdown 模板（`.md` 文件）也很好用，但：

* 好处：

  * 简单、兼容老版本；
  * 对纯技术团队来说完全够用。
* 缺点：

  * 无法强制校验必填项（大家经常只改一行标题就点提交）；
  * UI 不够直观，尤其对产品、运营等非技术角色不够友好。

所以，如果你是给 **团队内部用、且想提升规范**，**YAML 表单更适合**。
如果只是个人项目、或者团队非常小，Markdown 模板已经足够。

---

## 常见问题与注意事项

### 1. 模板没生效怎么办？

检查这些点：

* 文件路径是否正确：
  必须是 `.github/ISSUE_TEMPLATE/xxx.yml` 或 `.github/ISSUE_TEMPLATE/xxx.md`
* 分支是否正确：
  模板必须在默认分支（`main` / `master`）上才会生效；
* 文件名大小写：
  GitHub 对大小写是敏感的，`ISSUE_TEMPLATE` 目录名一定要对。

### 2. 改了模板但页面没变化？

* 浏览器可能有缓存，试着刷新 / 无痕窗口打开；
* 确认代码已经 `push` 到 GitHub；
* 如果你是 Fork 仓库，模板是跟着当前 repo 走的，不会继承上游仓库的模板。

### 3. 可以为组织统一配置模板吗？

* 可以在 **组织级别的 `.github` 仓库** 中配置默认模板，这样组织内的仓库如果本身没有模板，就会使用组织模板。
* 但这属于进阶玩法，这篇先不展开。

### 4. YAML 写错了怎么办？

* YAML 对缩进和空格比较敏感；
* 如果写错，有时 GitHub 会直接无视这个模板 / 报错；
* 建议：

  * 使用编辑器的 YAML 高亮和校验（VS Code 非常好用）；
  * 保证缩进是空格，且层级一致。

---

## 最佳实践与建议

1. **明确目标：先从一个「新需求模板」开始**，不要一上来就搞一堆复杂配置。
2. **强制填写核心字段**：背景、需求描述、验收标准，至少这三项建议必填。
3. **统一标题前缀**：比如 `[需求]` / `[Bug]`，方便筛选和搜索。
4. **自动打标签**：减少后续手动维护，比如 `feature`、`bug`、`enhancement`。
5. **适度即可**：模板太长、太复杂，用户会烦；保持在「引导清晰，又不至于太啰嗦」的平衡点。
6. **定期回顾**：用一两个月后，回头看看：

   * 哪些字段大家从来不填 → 可以删；
   * 哪些信息总是缺 → 加一个字段。

---

## 小结 / 结论

这篇文章里我们做了几件事：

* 搞清楚了 **Issue 模板 / YAML 表单 / Markdown 模板** 这些核心概念；
* 实际配置了一套 **「新需求 / Feature」表单模板** 和一个可选的 Bug 模板；
* 用步骤和命令跑完了 **从创建目录 → 写模板 → 推送 → 验证** 的完整流程；
* 解释了为什么推荐用 YAML 表单，以及常见的配置坑。

如果你把这些步骤在自己的仓库跑一遍，你的团队提需求这件事，质量会立刻有肉眼可见的提升——至少从“一句废话 Issue”变成了“可读、可执行的需求描述”。

---

## 参考与延伸阅读

你可以在这些关键词下继续查官方文档和示例：

* GitHub Docs：Issue and pull request templates
* 关键词：

  * `github issue template yaml`
  * `github issue forms`
  * `github .github/ISSUE_TEMPLATE examples`

（如果你团队同时用 Gitea & GitHub，其实两边的理念是通的，配置方式也很接近。）

---

## 元信息

* **预计阅读时长**：8–12 分钟
* **标签**：GitHub、协作效率、Issue 模板、团队规范、需求管理
* **SEO 关键词**：

  * GitHub Issue 模板
  * GitHub Issue Template 配置
  * YAML Issue Form 教程
  * Feature Request 模板
* **元描述（Meta Description）**：
  本文详细介绍如何在 GitHub 仓库中配置 Issue 模板，尤其是用于新需求 / Feature 的 YAML 表单模板和 Bug 模板，包含完整目录结构、配置示例、常见问题与最佳实践，帮助团队规范需求提交流程，提升协作效率。

---

## 行动号召（CTA）

如果你已经看完了，我建议你现在就：

1. 找一个你常用的 GitHub 仓库；
2. 按文中步骤创建 `.github/ISSUE_TEMPLATE/feature-request.yml`；
3. 推上去，自己开一个测试 Issue 感受一下效果。

