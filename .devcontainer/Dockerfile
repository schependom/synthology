FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

# Install unixODBC (Required for the dlv.dlv.x86-64-linux-elf-unixodbc version)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y unixodbc unixodbc-dev \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install DLV
# Download directly to /usr/local/bin so it's in PATH
RUN curl -o /usr/local/bin/dlv https://www.dlvsystem.it/files/dlv.x86-64-linux-elf-unixodbc \
    && chmod +x /usr/local/bin/dlv
