---
title: "Docker 常用使用教程：从入门到 Compose 实战（含 save/load 与权限排障）"
date: 2026-02-12
draft: false
categories: ["linux", "devops"]
tags: ["docker", "docker-compose", "container", "镜像管理", "权限排障"]
description: "一篇覆盖 Docker 常用命令、Dockerfile、Compose、镜像 save/load 离线迁移与挂载权限排障的实战教程。"
keywords: ["Docker 教程", "Docker Compose", "docker save", "docker load", "UID GID", "bind mount 权限"]
---

### **标题**

Docker 常用使用教程：从入门到 Compose 实战（含 save/load 与权限排障）

---

### **副标题 / 摘要**

很多人会 `docker run`，但一到离线交付和挂载权限就卡住。  
本文按“能直接落地”的顺序，带你走完 Docker 常用命令、Dockerfile、Compose、`save/load`，以及最常见的 UID/GID 权限问题。

---

### **目标读者**

* 想系统掌握 Docker 日常用法的开发者
* 需要用 Docker Compose 跑本地或测试环境的工程师
* 经常遇到“挂载目录写不进去”权限问题的人

---

### **背景 / 动机**

在“我的机器能跑、你的机器跑不起来”的场景里，Docker 的价值不是概念，而是交付稳定性。  
但实际使用中，常见断点通常在三处：

1. 命令会用，但不知道整套流程怎么串起来
2. 需要离线迁移时，不清楚 `save/load` 怎么和 Compose 配合
3. 挂载宿主目录后，容器用户与宿主目录属主不一致导致 `Permission denied`

---

### **核心概念**

* **镜像（Image）**：应用运行模板，分层存储，可复用
* **容器（Container）**：镜像的运行实例
* **仓库（Registry）**：镜像分发中心（如 Docker Hub）
* **卷（Volume）**：由 Docker 管理的数据持久化目录
* **Bind Mount**：把宿主机目录直接挂载进容器
* **Compose**：用一个 `compose.yaml` 管理多容器应用

---

## 一、安装与最小验证

先确保 Docker CLI 与 daemon 可用：

```bash
docker --version
docker info
docker run --rm hello-world
```

如果 `hello-world` 能成功输出欢迎信息，说明基础环境已就绪。

---

## 二、镜像常用命令（必会）

```bash
# 拉取镜像
docker pull nginx:1.27

# 查看本地镜像
docker images

# 给镜像打标签
docker tag nginx:1.27 my-registry.local/nginx:1.27

# 推送镜像（需先 docker login）
docker push my-registry.local/nginx:1.27

# 删除镜像
docker rmi nginx:1.27
```

建议不要在生产场景依赖 `latest`，固定版本更可控。

---

## 三、容器常用命令（必会）

```bash
# 后台启动容器并映射端口
docker run -d --name web -p 8080:80 nginx:1.27

# 查看运行中的容器
docker ps

# 查看全部容器（含已退出）
docker ps -a

# 查看日志
docker logs -f web

# 进入容器
docker exec -it web sh

# 停止 / 启动 / 重启
docker stop web
docker start web
docker restart web

# 删除容器
docker rm -f web
```

---

## 四、数据持久化：Volume 与 Bind Mount

### 1）Volume（推荐默认）

```bash
docker volume create app_data
docker run -d --name db \
  -e POSTGRES_PASSWORD=secret \
  -v app_data:/var/lib/postgresql/data \
  postgres:16
```

优点是和宿主目录权限耦合更少，迁移与备份也更稳定。

### 2）Bind Mount（开发常用）

```bash
docker run -d --name app \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  myapp:1.0.0
```

优点是直观；缺点是最容易踩 UID/GID 权限坑（后面专门讲）。

---

## 五、Dockerfile 最小可用模板

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

构建和运行：

```bash
docker build -t myapp:1.0.0 .
docker run -d --name myapp -p 8000:8000 myapp:1.0.0
```

---

## 六、Docker Compose 实战（单机多服务）

示例：`app + redis`。

```yaml
services:
  app:
    image: myapp:1.0.0
    container_name: myapp
    ports:
      - "8000:8000"
    environment:
      REDIS_HOST: redis
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7.2
    container_name: myredis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

常用命令：

```bash
docker compose up -d
docker compose ps
docker compose logs -f app
docker compose down
```

---

## 七、`docker save/load` 怎么写（含 Compose 离线交付）

你可以把这节理解成“镜像搬家流程”。

### 场景 A：单个镜像离线迁移

```bash
# 导出镜像
docker pull nginx:1.27
docker save -o nginx_1.27.tar nginx:1.27

# 导入镜像
docker load -i nginx_1.27.tar
```

### 场景 B：Compose 应用离线交付（重点）

`docker compose` 本身没有 `save/load` 子命令。  
做法是：先列出 Compose 用到的镜像，再用 `docker save/load` 处理这些镜像。

```bash
# 列出 compose.yaml 中所有镜像
docker compose config --images

# 确保本地都有这些镜像
docker compose config --images | xargs -r docker pull

# 一次性打包为 tar
docker save -o stack-images.tar $(docker compose config --images)
```

目标机器导入并启动：

```bash
docker load -i stack-images.tar
docker compose up -d --no-build
```

> 说明：`--no-build` 可避免目标机器误触发本地构建，保证按导入镜像启动。

---

## 八、常见权限问题：挂载目录与 Dockerfile 用户不一致

你提到的核心问题就是这一类。

### 1）典型现象

* 容器能启动，但写日志/上传文件时报 `Permission denied`
* 应用在容器内创建目录失败

### 2）根因（关键）

Bind mount 时，容器看到的是宿主机目录真实 UID/GID。  
如果 Dockerfile 里 `USER` 是 `1001:1001`，而宿主目录属主是 `1000:1000`，容器用户就可能无写权限。

### 3）排查命令

```bash
# 宿主机查看目录属主属组（数字形式）
ls -ln ./data

# 容器内查看当前用户
docker exec -it myapp sh -c 'id && ls -ln /app/data'
```

### 4）推荐修复方案（按优先级）

#### 方案 A：统一 UID/GID（推荐）

Dockerfile：

```dockerfile
FROM python:3.12-slim

ARG APP_UID=1001
ARG APP_GID=1001

RUN groupadd -g ${APP_GID} app && useradd -m -u ${APP_UID} -g ${APP_GID} app

WORKDIR /app
COPY . .
RUN chown -R app:app /app

USER app
CMD ["python", "app.py"]
```

Compose：

```yaml
services:
  app:
    build:
      context: .
      args:
        APP_UID: "${APP_UID:-1001}"
        APP_GID: "${APP_GID:-1001}"
    user: "${APP_UID:-1001}:${APP_GID:-1001}"
    volumes:
      - ./data:/app/data
```

宿主机目录对齐：

```bash
sudo chown -R 1001:1001 ./data
```

#### 方案 B：开发机直接跟随当前用户

```bash
export APP_UID=$(id -u)
export APP_GID=$(id -g)
docker compose up -d --build
```

#### 方案 C：入口脚本启动时 `chown`（仅小目录）

适合开发调试，不建议在大目录或高频重启场景长期使用，会拖慢启动。

---

## 九、常见问题与注意事项

1. `permission denied while trying to connect to the Docker daemon socket`  
   这是 docker socket 权限问题；和本文的 bind mount 文件权限问题不是一回事。
2. 端口冲突：`bind: address already in use`  
   改端口映射或释放占用端口。
3. 容器反复退出  
   先看 `docker logs <container>`，再看 `docker inspect` 的 `State` 字段。
4. 磁盘占用暴涨  
   用 `docker system df` 先定位，再谨慎执行 `docker system prune`。

---

## 十、最佳实践清单

* 镜像版本固定，不依赖 `latest`
* Dockerfile 分层合理，减少无效重建
* 优先非 root 用户运行容器
* Compose 中显式声明 `restart`、`volumes`、`environment`
* 对 bind mount 提前约定 UID/GID，避免上线后才排权限问题
* 离线交付用 `save/load` 固化镜像集，减少环境波动

---

## 小结 / 结论

把 Docker 用顺，关键是掌握“整链路”而不只是命令记忆：  
**构建镜像 -> 运行容器 -> Compose 编排 -> save/load 交付 -> 权限排障**。  
其中，挂载目录写不进去几乎都能回到 UID/GID 对齐这个根因上。

---

## 参考与延伸阅读

* https://docs.docker.com/engine/
* https://docs.docker.com/reference/cli/docker/image/save/
* https://docs.docker.com/reference/cli/docker/image/load/
* https://docs.docker.com/compose/
* https://docs.docker.com/engine/storage/bind-mounts/

---

## 元信息

* 阅读时长：约 12 分钟
* 关键词：Docker 教程、Docker Compose、save/load、挂载权限、UID/GID
* 适用场景：本地开发、测试环境交付、离线部署

---

## 行动号召（CTA）

把你当前的 `Dockerfile` 和 `compose.yaml` 发出来，我可以按你的实际目录结构给一版“UID/GID 不踩坑”的可直接运行配置。
