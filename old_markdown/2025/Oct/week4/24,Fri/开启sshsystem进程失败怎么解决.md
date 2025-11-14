**标题：**
🚀 在无 sudo 环境下让 sshd 常驻运行：从错误排查到 nohup 与 systemd 双方案实战

**副标题 / 摘要：**
本文讲述如何在普通用户权限下运行 OpenSSH 服务，逐步解决“连接被拒绝”“密码认证失败”“systemd start-limit-hit”等典型问题，并最终用 nohup 与 systemd 实现持久运行。

**目标读者：**
Linux 中级用户、科研或企业多用户服务器使用者、无 root 权限的 SSH 自部署者。

---

## 一、背景 / 动机

在部分高校实验室或云主机环境中，普通账户没有 sudo 权限，默认 sshd 服务无法启动。
当我们需要：

* 远程访问自己的 Linux 主机；
* 使用 VS Code Remote 或 SCP 传文件；
* 但又无法修改系统级配置；
  就必须在**用户态**自行运行 sshd。
  然而这会引发一系列问题：端口冲突、防火墙、认证失败、`start-limit-hit` 等。

---

## 二、核心概念

| 名称                  | 含义                            |
| ------------------- | ----------------------------- |
| **sshd**            | OpenSSH 守护进程，负责处理 SSH 登录请求    |
| **用户态 sshd**        | 非 root 用户手动启动的 sshd 实例，仅有用户权限 |
| **authorized_keys** | 存放允许登录的公钥                     |
| **nohup**           | 让程序脱离终端后台运行                   |
| **systemd --user**  | 用户级 systemd 实例，可管理自启服务        |
| **start-limit-hit** | systemd 检测到服务频繁退出后自动暂停重启      |

---

## 三、实践指南 / 全流程步骤

### 1️⃣ 生成并配置 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "" -f ~/.ssh/id_ed25519_noemail
cat ~/.ssh/id_ed25519_noemail.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

确保 `~/.ssh/authorized_keys` 权限正确。

---

### 2️⃣ 编写用户态 sshd 配置文件

`~/.ssh/ssh_config_pub`

```bash
Port 2223
ListenAddress 0.0.0.0
HostKey /home/chenhm/.ssh/ssh_host_ed25519_key
AuthorizedKeysFile /home/chenhm/.ssh/authorized_keys
PasswordAuthentication no
PubkeyAuthentication yes
PidFile /home/chenhm/.ssh/sshd_pub.pid
LogLevel INFO
SyslogFacility AUTH
```

生成 HostKey：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ssh_host_ed25519_key -N ""
```

---

### 3️⃣ 手动调试启动

```bash
/usr/bin/sshd -d -f ~/.ssh/ssh_config_pub
```

看到
`Server listening on 0.0.0.0 port 2223`
即成功。

---

## 四、两种常驻运行方案

### ✅ 方案 A：使用 nohup 后台运行（最简单）

```bash
nohup /usr/bin/sshd -f ~/.ssh/ssh_config_pub -E ~/.ssh/sshd_pub.log >/dev/null 2>&1 &
```

* 退出终端后依旧运行；
* 查看进程：

  ```bash
  ps -ef | grep "sshd -f"
  ```
* 查看日志：

  ```bash
  tail -f ~/.ssh/sshd_pub.log
  ```
* 停止：

  ```bash
  pkill -f "sshd -f /home/chenhm/.ssh/ssh_config_pub"
  ```

**优点：** 无依赖、立即可用。
**缺点：** 系统重启后不会自动恢复。

---

### ✅ 方案 B：使用 systemd 用户服务（自动重启 / 开机自启）

#### 1️⃣ 创建服务文件

`~/.config/systemd/user/sshd-user.service`

```ini
[Unit]
Description=User-level SSH server

[Service]
Type=forking
ExecStart=/usr/bin/sshd -f /home/chenhm/.ssh/ssh_config_pub -E /home/chenhm/.ssh/sshd_pub.log
PIDFile=/home/chenhm/.ssh/sshd_pub.pid
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

#### 2️⃣ 加载并启动

```bash
systemctl --user daemon-reload
systemctl --user enable sshd-user
systemctl --user start sshd-user
```

#### 3️⃣ 验证

```bash
systemctl --user status sshd-user
ss -tlnp | grep sshd
```

出现
`Active: active (running)` 与 `0.0.0.0:2223` 即为成功。

---

## 五、错误排查与解决历程

| 错误                                                | 原因                           | 解决方案                                        |
| ------------------------------------------------- | ---------------------------- | ------------------------------------------- |
| `Connection refused`                              | sshd 未监听公网 / 端口被防火墙拦截        | 改 `ListenAddress 0.0.0.0`，检查 `ss -tlnp`     |
| `Permission denied (password)`                    | 非 root 无法访问 `/etc/shadow` 密码 | 使用 公钥登录                                     |
| `Bind to port ... failed: Address already in use` | 端口被旧 sshd 占用                 | `pkill -f "sshd -f"`                        |
| `start-limit-hit`                                 | systemd 认为服务频繁退出             | 在 service 文件中加入 `Type=forking` 与 `PIDFile=` |
| 无日志输出                                             | 路径错误或权限不够                    | 使用 `-E ~/.ssh/sshd.log` 输出日志                |

---

## 六、为什么这样做可行

* **用户态 sshd** 不需要 root ，因为它只监听用户有权限的端口（≥1024）。
* **公钥认证** 绕过 `/etc/shadow` 权限限制。
* **Type=forking** 让 systemd 正确识别后台 daemon。
* **PIDFile** 帮助 systemd 追踪进程。

---

## 七、常见注意事项

1. **端口 > 1024**：非 root 用户无法绑定低端口。
2. **防火墙**：需放行对应端口，否则外部连接被拒。
3. **权限严格**：`~/.ssh` 必须 700， `authorized_keys` 必须 600。
4. **重复实例**：不同配置文件需独立 PidFile 与 日志文件。
5. **开机自启**：启用 `systemctl --user enable sshd-user` 后即可。

---

## 八、最佳实践与建议

* 用 **nohup** 调试、临时运行；
* 用 **systemd --user** 管理正式常驻；
* 公网接口建议只开放密钥登录；
* 不同端口区分内网与外网访问；
* 结合 crontab @reboot 可在 systemd 不可用时兜底启动。

---

## 九、小结 / 结论

本文从零开始在无 sudo 环境下部署 SSH 服务：

1. 生成密钥并启用公钥认证；
2. 编写用户态 sshd 配置；
3. 先用 nohup 验证，再用 systemd 稳定自启；
4. 排查并修复了 `start-limit-hit` 、端口冲突、认证失败等问题。

最终实现了：

* 多端口多实例；
* 自动重启；
* 开机自启；
* 安全可靠的远程访问。

---

## 十、参考与延伸阅读

* [OpenSSH 官方手册](https://man.openbsd.org/sshd.8)
* [systemd User Services 文档](https://wiki.archlinux.org/title/Systemd/User)
* [OpenSSH Key Management](https://www.ssh.com/academy/ssh/keygen)

---

**元信息**

* 阅读时长：约 10 分钟
* 标签：`SSH`、`Linux`、`systemd`、`nohup`、`无sudo`
* SEO 关键词：`无sudo sshd systemd 用户态 OpenSSH 启动失败 start-limit-hit`
* 元描述：解决无 sudo 权限下 OpenSSH 启动失败的完整实战指南。

---

**行动号召（CTA）**
💡 试试在你的实验室服务器上部署一个用户级 sshd 吧！
如果觉得本文有帮助，欢迎收藏、分享或在评论区交流你的坑与经验 🚀

