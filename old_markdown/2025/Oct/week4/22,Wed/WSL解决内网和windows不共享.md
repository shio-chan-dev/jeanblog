# 📝 Windows + WSL2 端口转发教程（访问 Flask 5000）

## 前提条件

1. 你正在使用 **WSL2**（Ubuntu 或其他 Linux 发行版）
2. Windows 主机能访问局域网（Wi-Fi 或以太网）
3. Flask 服务在 WSL2 中运行，并监听：

```python
app.run(host="0.0.0.0", port=5000)
```

> ⚠️ `host="0.0.0.0"` 必须，否则外部无法访问

---

## 第 1 步：确认 WSL2 的 IP

在 WSL2 中运行：

```bash
ip addr show eth0
```

你会看到类似：

```
inet 172.26.209.37/20
```

> 记下 `inet` 后面的 IP（本例是 `172.26.209.37`），这是 WSL2 内部 IP。

---

## 第 2 步：打开 PowerShell（管理员模式）

1. 按 `Win + X` → 选择 **Windows PowerShell (管理员)**
2. 确认管理员权限，必要时允许 UAC 提示

---

## 第 3 步：设置端口转发

在 PowerShell 中执行以下命令，将 Windows 的 5000 端口转发到 WSL2：

```powershell
# 将 Windows 5000 端口转发到 WSL2 的 5000
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=172.26.209.37

# 开放防火墙，让局域网可以访问
netsh advfirewall firewall add rule name="WSL Flask 5000" dir=in action=allow protocol=TCP localport=5000
```

* `listenaddress=0.0.0.0` 表示监听 Windows 所有网卡（局域网可访问）
* `connectaddress=172.26.209.37` 是 WSL2 内部 IP
* 防火墙规则允许外部设备访问 Windows 5000 端口

---

## 第 4 步：测试端口转发

1. **在 Windows 本机浏览器或 curl** 测试：

```powershell
curl http://localhost:5000
# 或者
curl http://192.168.1.227:5000
```

2. **在局域网设备上访问**：

```text
http://<Windows局域网IP>:5000
```

> 示例：`http://192.168.1.227:5000`

---

## 第 5 步（可选）：自动更新脚本

WSL2 IP 每次重启可能变化，为了自动更新转发规则，可创建 PowerShell 脚本 `wsl_port_forward.ps1`：

```powershell
# 获取当前 WSL IP
$wsl_ip = wsl hostname -I | ForEach-Object { $_.Split(" ")[0] }
Write-Host "Detected WSL IP: $wsl_ip"

# 删除旧规则
netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=0.0.0.0

# 添加新规则
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=$wsl_ip

# 放行防火墙
netsh advfirewall firewall add rule name="WSL Flask 5000" dir=in action=allow protocol=TCP localport=5000
```

* 保存脚本，**每次 WSL 启动前执行**即可
* 自动检测当前 WSL IP，更新端口转发规则

---

## 第 6 步：注意事项

1. Flask 必须监听 `0.0.0.0`，否则只能本机访问
2. 确保 Windows 防火墙允许 TCP 5000 端口
3. 如果局域网设备仍无法访问：

   * 检查路由器是否阻止局域网内端口访问
   * 检查 Windows 防火墙是否生效
4. WSL2 NAT 模式下，局域网不能直接访问 WSL 内部 IP，只能通过 Windows IP + 转发端口访问

---

## ✅ 总结

* WSL2 默认网络隔离，局域网无法直接访问
* 通过 **Windows 端口转发 + 防火墙放行**，局域网设备可以访问 WSL2 中的 Flask 服务
* 自动化脚本可以解决 WSL2 重启后 IP 变化的问题

