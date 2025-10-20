FROM python:3.12-slim

WORKDIR /app/assignment2

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install "uvicorn[standard]" fastapi python-multipart numpy pillow

# 复制代码和模型
COPY assignment2/ ./

# 训练模型（如果还没有预训练模型的话）
RUN python train.py

# 暴露API端口
EXPOSE 8000

# 设置启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]