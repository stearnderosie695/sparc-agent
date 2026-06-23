FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade "pip==23.3.1" "setuptools==68.2.2" "wheel==0.41.3"

WORKDIR /opt/smarg
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN python -m pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["darkroom"]
CMD ["--help"]
