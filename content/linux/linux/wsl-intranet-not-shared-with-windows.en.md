---
title: "Expose WSL2 Services to the LAN via Windows Port Forwarding"
date: 2025-10-22
draft: false
---

# Windows + WSL2 Port Forwarding Guide (Access Flask 5000)

## Prerequisites

1. You are using **WSL2** (Ubuntu or another Linux distro)
2. The Windows host can access the LAN (Wi-Fi or Ethernet)
3. A Flask service is running inside WSL2 and listening on:

```python
app.run(host="0.0.0.0", port=5000)
```

> `host="0.0.0.0"` is required; otherwise external access will fail.

---

## Step 1: Check the WSL2 IP

In WSL2:

```bash
ip addr show eth0
```

You should see something like:

```
inet 172.26.209.37/20
```

> Record the IP after `inet` (here: `172.26.209.37`). This is the WSL2 internal IP.

---

## Step 2: Open PowerShell (Admin)

1. Press `Win + X` and select **Windows PowerShell (Admin)**
2. Confirm admin privileges if prompted by UAC

---

## Step 3: Add Port Forwarding

In PowerShell, forward Windows port 5000 to WSL2:

```powershell
# Forward Windows port 5000 to WSL2 port 5000
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=172.26.209.37

# Allow LAN access through the firewall
netsh advfirewall firewall add rule name="WSL Flask 5000" dir=in action=allow protocol=TCP localport=5000
```

- `listenaddress=0.0.0.0` listens on all Windows interfaces
- `connectaddress=172.26.209.37` is the WSL2 internal IP
- The firewall rule allows LAN devices to access Windows port 5000

---

## Step 4: Test the Forwarding

1. **On the Windows machine:**

```powershell
curl http://localhost:5000
# or
curl http://192.168.1.227:5000
```

2. **From another LAN device:**

```
http://<Windows-LAN-IP>:5000
```

Example:

```
http://192.168.1.227:5000
```

---

## Step 5 (Optional): Auto-update Script

WSL2 IP can change after reboot. You can create a PowerShell script `wsl_port_forward.ps1` to update rules:

```powershell
# Get current WSL IP
$wsl_ip = wsl hostname -I | ForEach-Object { $_.Split(" ")[0] }
Write-Host "Detected WSL IP: $wsl_ip"

# Remove old rule
netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=0.0.0.0

# Add new rule
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=$wsl_ip

# Allow firewall
netsh advfirewall firewall add rule name="WSL Flask 5000" dir=in action=allow protocol=TCP localport=5000
```

- Run this script before starting WSL each time
- It detects the current WSL IP and updates the forwarding rule

---

## Step 6: Notes

1. Flask must listen on `0.0.0.0`, otherwise only local access works
2. Ensure Windows Firewall allows TCP port 5000
3. If LAN devices still cannot access:
   - Check router policies for blocked LAN ports
   - Verify Windows firewall rules
4. WSL2 uses NAT; LAN devices cannot reach the WSL IP directly. Use Windows IP + port forwarding

---

## Summary

- WSL2 networking is isolated by default; LAN cannot access WSL directly
- Windows port forwarding + firewall rules enable LAN access to WSL services
- An auto-update script can handle WSL IP changes after reboot
