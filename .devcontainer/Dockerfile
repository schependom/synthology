# Dockerfile for Synthology UNIX development container
# with Python, uv, and DLV installed
#
# Author: Vincent Van Schependom

FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install DLV
#   -> Download directly to /usr/local/bin so it's in PATH
#   -> The post_create.sh script will update the config file to point to this location
RUN curl -o /usr/local/bin/dlv https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-static.bin \
    && chmod +x /usr/local/bin/dlv
