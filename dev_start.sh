#!/bin/bash

# 配置区域
CONTAINER_NAME="sarax_dev"
COMPOSE_PATH="$HOME/sarax_dev/docker-compose.yml"
WORKSPACE_DIR="/home/user/sarax_ws/src/sarax" # 容器内的项目目录
ROOT_PATH="/home/user/sarax_ws"
# 1. 检查 Docker 服务是否启动
if ! systemctl is-active --quiet docker; then
    echo "❌ Docker 服务未运行，请先启动 Docker: sudo systemctl start docker"
    exit 1
fi

# 2. 授权 GUI 显示权限 (针对 ROS/Rviz 必选)
if [ -n "$DISPLAY" ]; then
    xhost +local:docker > /dev/null
    echo "🖥️  已授权本地 Docker 访问 X11 图形界面"
fi

# 3. 检测容器状态
# status=running 过滤出真正跑起来的容器
IS_RUNNING=$(docker ps -q -f name=^/${CONTAINER_NAME}$ -f status=running)

if [ -z "$IS_RUNNING" ]; then
    echo "🚀 正在启动开发环境 (docker compose)..."
    # 使用 -f 指定路径，确保在任何地方都能启动
    docker compose -f "$COMPOSE_PATH" up -d
    
    # 给予一点缓冲时间让 Entrypoint 脚本执行完毕
    sleep 1
else
    echo "✅ 开发环境已在运行中"
fi

# 4. 进入容器并自动切换到 workspace 目录
echo "📂 正在进入容器终端..."
docker exec -it -w "$WORKSPACE_DIR" "$CONTAINER_NAME" bash
docker exec -it -w ""
