# 在局域网访问 Windows WSL2 上的 Git Bare 仓库

在开发中，我们经常需要在多台电脑之间共享 Git 仓库。如果你在 Windows 上使用 WSL2，并且想在同一局域网的其他电脑访问 WSL2 上的 Git bare 仓库，本文将一步步教你实现。

---

## 1. 在 WSL2 创建 Git Bare 仓库

打开 WSL2 终端，进入你想存放仓库的目录，执行：

```bash
git init --bare my_project.git
```

* `my_project.git` 是 bare 仓库，不含工作区，仅用于推送和拉取。
* bare 仓库就像远程仓库一样，可以被克隆和操作。

---

## 2. 配置 WSL2 的 SSH 服务

为了让其他电脑访问仓库，需要通过 SSH 访问 WSL2。

1. 安装 SSH 服务：

```bash
sudo apt update
sudo apt install openssh-server -y
```

2. 启动 SSH 服务：

```bash
sudo service ssh start
```

3. 检查 SSH 服务状态：

```bash
sudo service ssh status
```

4. 默认端口是 22，可以在 `/etc/ssh/sshd_config` 修改。

---

## 3. 获取 WSL2 IP 地址

在 WSL2 终端运行：

```bash
ip addr
```

找到 `eth0` 下的 `inet` 地址，例如：

```
inet 172.25.190.21/20
```

> 注意：WSL2 IP 每次重启可能变化。

---

## 4. 配置 Windows 防火墙

为了让局域网电脑访问，需要允许 SSH 端口通过防火墙。

1. 打开 **Windows 防火墙 → 高级设置 → 入站规则 → 新建规则**
2. 规则类型选择 **端口** → TCP → 指定端口（22 或自定义端口如 2222）
3. 允许连接 → 应用到 **域/专用/公用**
4. 给规则命名 → 完成

---

## 5. 推荐：使用端口转发解决 WSL2 IP 变化问题

因为 WSL2 IP 会变，推荐使用 Windows 端口转发：

1. 打开 PowerShell（管理员），执行：

```powershell
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=<WSL_IP>
```

2. 然后从局域网其他电脑通过 **Windows IP + 2222** 访问 WSL2：

```bash
git clone ssh://user@WINDOWS_IP:2222/home/user/my_project.git
```

* `user` 是 WSL2 用户名
* `WINDOWS_IP` 是 Windows 主机在局域网的 IP

---

## 6. 从其他电脑克隆、推送和拉取

克隆仓库：

```bash
git clone ssh://user@WINDOWS_IP:2222/home/user/my_project.git
```

提交修改：

```bash
git add .
git commit -m "修改说明"
git push origin main  # 或 master
```

拉取更新：

```bash
git pull origin main
```

---

## 7. 总结

1. WSL2 自身有虚拟网络，IP 每次启动可能变化。
2. 使用 **端口转发 + 防火墙放行** 是最稳妥的方式。
3. bare 仓库在 WSL2 内部创建，其他电脑就像访问远程仓库一样操作。

通过上述步骤，你就可以在局域网内多台电脑访问 WSL2 上的 Git 仓库，轻松实现代码共享和协作。

