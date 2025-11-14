# 🚀 本地搭建 Gitea：打造你的私人 GitHub（含已有仓库导入指南）

**副标题 / 摘要：**
本文将手把手教你在本地电脑上安装轻量级 Git 服务器 —— Gitea。
无需 root、不会影响系统环境，让你像在 GitHub 一样管理、查看和推送项目，还能导入已有仓库。

**目标读者：**
👉 适合个人开发者、独立工程师、小型团队技术负责人。
适用于初中级开发者，有 Git 基础即可上手。

---

## 🧠 背景 / 动机

很多开发者希望：

* 在公司电脑或内网环境下托管代码；
* 不想使用云端（如 GitHub、Gitee）；
* 又希望有 Web 界面、Pull Request、代码浏览体验。

但 **GitLab 太重**（动辄占用数 GB 内存），而 Gitea 则：

> 🌱 轻量级、单可执行文件、支持 PR、Wiki、Issue、CI/CD。

只需几分钟，你就能拥有一个完全属于自己的“小型 GitHub”。

---

## 📘 核心概念

| 名称               | 说明                        |
| ---------------- | ------------------------- |
| **GitLab**       | 功能最强大的开源 Git 平台，但资源占用高    |
| **Gitea**        | 轻量级自托管 Git 服务，界面类似 GitHub |
| **Bare 仓库**      | 只保存版本数据、不包含工作区的纯仓库        |
| **Pull Request** | 一个分支向另一个分支发起的合并请求         |
| **SQLite**       | Gitea 默认使用的轻量数据库，无需额外配置   |

---

## 🧩 实践指南 / 安装步骤

### 1️⃣ 准备环境

系统要求：Linux / macOS / Windows 均可
推荐配置：内存 ≥ 512MB，磁盘 ≥ 1GB

### 2️⃣ 创建目录并下载 Gitea

```bash
mkdir -p ~/gitea
cd ~/gitea
wget -O gitea https://dl.gitea.io/gitea/1.22.0/gitea-1.22.0-linux-amd64
chmod +x gitea
```

### 3️⃣ 启动 Gitea

```bash
./gitea web --port 3000
```

浏览器访问：[http://localhost:3000](http://localhost:3000)

### 4️⃣ 安装引导

在页面中填写：

* 数据库类型：`SQLite3`
* 仓库根路径：`/home/<username>/gitea/repos`
* Gitea Base URL：`http://localhost:3000`
* 创建管理员账号

---

## 💻 可运行示例：推送已有仓库

假设你的本地项目位于 `/home/gong/projects/scrapy`：

1️⃣ 在 Gitea 上创建新仓库 `scrapy`
2️⃣ 在项目目录中执行：

```bash
cd ~/projects/scrapy
git remote set-url origin http://localhost:3000/JeanphiloGong/scrapy.git
git push -u origin --all
git push -u origin --tags
```

刷新网页，你会看到完整的项目历史出现在 Gitea 界面。

---

## 怎么注册到系统服务

### 1.前提准备
假设安装在
```swift
/home/gong/gitea
```

可执行文件路径在:
```swift
/home/gong/gitea/gitea
```
运行用户 `gong`
不要使用root用户运行gitea



### ⚙️ 二、创建 Gitea 的 systemd 服务文件

1️⃣ 打开或创建服务文件：
```bash
sudo nano /etc/systemd/system/gitea.service
```
2️⃣ 粘贴以下配置内容（适用于单用户本地部署）：

```ini
[Unit]
Description=Gitea (Self-hosted Git Service)
After=network.target

[Service]
# 运行用户和组
User=gong
Group=gong

# Gitea 工作目录（你的 gitea 程序所在目录）
WorkingDirectory=/home/gong/gitea

# 启动命令
ExecStart=/home/gong/gitea/gitea web --config /home/gong/gitea/custom/conf/app.ini

# 自动重启策略
Restart=always
RestartSec=10s

# 环境变量（可选）
Environment=USER=gong HOME=/home/gong GITEA_WORK_DIR=/home/gong/gitea

# 限制权限（安全）
PrivateTmp=true
ProtectSystem=full
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```


✅ 说明：

WorkingDirectory 是你运行 Gitea 的目录；

ExecStart 指定启动命令；

Restart=always 确保意外中断后自动重启。

### 🔁 三、加载并启用服务
```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 设置开机自启
sudo systemctl enable gitea

# 启动服务
sudo systemctl start gitea

# 查看运行状态
sudo systemctl status gitea
```


你应该能看到：

Active: active (running)

### 🧠 四、日志查看命令

查看实时日志：

sudo journalctl -u gitea -f


查看历史日志：

sudo journalctl -u gitea --since "1 hour ago"


## ⚙️ 解释与原理

Gitea 是一个基于 Go 语言开发的 **自托管 Git 服务**。
它的核心原理是直接管理本地的 Git 仓库目录（`~/gitea/repos`），
通过 HTTP/SSH 协议提供与 GitHub 相同的操作接口。

相比之下：

* `git init --bare` 是最原始的 Git 服务器，只能存代码；
* Gitea 在此基础上增加了 Web 界面、用户系统、PR、Wiki 等。

---

## ⚠️ 常见问题与注意事项

| 问题          | 原因             | 解决方案                          |
| ----------- | -------------- | ----------------------------- |
| 端口被占用（3000） | 系统已有服务占用       | 改用 `./gitea web --port 8080`  |
| 访问提示权限问题    | Gitea 以当前用户运行  | 检查仓库目录权限                      |
| 无法推送        | 仓库初始化冲突        | 不要勾选 “Initialize with README” |
| 推送慢或超时      | 使用 HTTP 而非 SSH | 配置 SSH key 后推送更快              |

---

## 🌟 最佳实践与建议

* 使用 `SQLite` 足够个人或小团队使用；
* 用 `nohup ./gitea web &` 后台运行；
* 定期备份目录：

  ```
  ~/gitea/repos/
  ~/gitea/data/gitea.db
  ~/gitea/custom/conf/app.ini
  ```
* 若将来扩展团队，可无缝迁移至公司服务器或 Docker。

---

## 🧾 小结 / 结论

本文带你从零开始完成：

1. 在本地电脑部署 Gitea
2. 修改端口运行不冲突
3. 将已有 Git 仓库推送到 Gitea
4. 拥有自己的 Web 界面、PR、历史记录

> 🎉 恭喜！你现在已经拥有一个完全属于自己的「私人 GitHub」。

---

## 🔗 参考与延伸阅读

* [Gitea 官方文档](https://docs.gitea.io/)
* [Gitea Releases 下载页](https://dl.gitea.io/gitea/)
* [Git 官方 Pro Book](https://git-scm.com/book/)
* [Forgejo - 社区维护的 Gitea 分支](https://forgejo.org/)

---

## 🧭 元信息

* **阅读时长：** 8 分钟
* **标签：** `Git`, `Gitea`, `自建服务`, `DevOps`, `版本控制`
* **SEO 关键词：** `Gitea 本地安装`, `自建 Git 服务器`, `私人 GitHub`, `导入本地仓库`
* **元描述：**
  在本地电脑上搭建轻量级 Git 服务器 Gitea，支持 PR、Web 浏览与仓库管理，不修改系统环境。

---

## 💬 行动号召（CTA）

试试看吧 👉

1. 打开终端运行安装命令；
2. 访问 [http://localhost:3000](http://localhost:3000)；
3. 创建你的第一个仓库；
4. 把项目推送上去！

💡 如果你想了解 **如何自动化启动 Gitea + 备份脚本**，欢迎在评论区留言，我会分享下一篇进阶文章。
