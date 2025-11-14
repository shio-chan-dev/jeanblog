# 在本地使用 Git 裸仓库实现开发环境和测试环境隔离

在全栈开发的过程中，我们常常遇到一个问题：**开发环境和测试环境如何隔离**？
很多人第一反应是用 GitHub 或 GitLab 来托管代码，但如果项目涉及隐私，不方便放在公共仓库，那该怎么办呢？

其实，Git 是分布式的，我们完全可以在 **本地电脑上建立一个“裸仓库 (bare repo)”**，当作“远程仓库”来用，从而实现 **开发环境 → 测试环境** 的代码迁移和同步。

---

## 什么是裸仓库 (bare repository)

* 普通 Git 仓库（`git init`）包含 **工作区 + .git 元数据**，可以直接编辑文件。
* 裸仓库（`git init --bare`）只有 Git 的版本信息，没有工作区，不能直接编辑文件，通常作为 **远程仓库** 来存储和同步代码。

简单理解：

* **开发仓库**：我在这里写代码。
* **裸仓库**：我用来存放代码历史，作为远程同步点。
* **测试仓库**：从裸仓库克隆出来，模拟运行环境。

---

## 步骤一：创建裸仓库

在本机某个目录（比如 `~/.repos`）下创建裸仓库：

```bash
mkdir -p ~/.repos
cd ~/.repos
git init --bare scrapy.git
```

这样你得到一个路径 `~/.repos/scrapy.git`，它就是本地的远程仓库。

---

## 步骤二：在开发仓库里添加远程

假设你的开发仓库在 `~/scrapy`：

```bash
cd ~/scrapy
git remote add local ~/.repos/scrapy.git
```

检查一下远程是否添加成功：

```bash
git remote -v
```

输出类似：

```
local	/home/gong/.repos/scrapy.git (fetch)
local	/home/gong/.repos/scrapy.git (push)
```

说明配置成功。

---

## 步骤三：推送代码到本地远程

将 `main` 分支推送到刚刚创建的裸仓库：

```bash
git push local main
```

这时裸仓库中已经保存了你所有的提交记录。

---

## 步骤四：在测试环境中克隆代码

假设你想在 `~/test-env` 下运行测试环境：

```bash
cd ~/test-env
git clone ~/.repos/scrapy.git
```

这样你就得到了一个干净的副本，可以在这里模拟部署、运行测试，而不会影响开发环境。

---
ps. 很多时候
```pgsql
warning: remote HEAD refers to nonexistent ref, unable to checkout
```
会出现这个错误,是由于我们新建的裸仓库虽然已经 init --bare 了,但是没有默认的HEAD指针,所以我们git clone的时候不知道该检出哪个分支

进入裸仓库,使用
```bash
cd ~/.repos/scrapy.git
git symbolic-ref HEAD refs/heads/main
```
之后再重新clone一次就可以了



## 步骤五：后续同步流程

* 在开发环境 (`~/scrapy`)：

  ```bash
  # 正常开发、提交
  git add .
  git commit -m "feat: 完成功能"

  # 推送到本地远程
  git push local main
  ```

* 在测试环境 (`~/test-env/scrapy`)：

  ```bash
  # 拉取最新代码
  git pull
  ```

这样你就能方便地在一台电脑上实现 **开发环境 → 测试环境** 的代码迁移和隔离。

---

## 总结

如果代码不方便上传到 GitHub/GitLab，完全可以通过本地裸仓库来实现前后端开发与测试环境的解耦。

优点：

* 不依赖外部平台，安全性高。
* 开发环境和测试环境隔离，互不干扰。
* 保留了完整的 Git 历史，方便版本管理。

后续如果项目规模扩大，也可以考虑引入 **私有 Git 服务（Gitea/GitLab CE）** 或 **Docker 部署**，进一步提升开发体验。

