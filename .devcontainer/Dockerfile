FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ARG APT_MIRROR=https://mirrors.aliyun.com
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i \
          -e "s|https\?://security.debian.org|${APT_MIRROR}|g" \
          -e "s|https\?://deb.debian.org|${APT_MIRROR}|g" \
          /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_INDEX_URL=${PIP_INDEX}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
