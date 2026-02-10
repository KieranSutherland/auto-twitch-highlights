FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_DEVICE=cuda

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    ca-certificates \
    libicu-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN pip install --no-cache-dir yt-dlp

# Install TwitchDownloaderCLI (properly)
RUN wget -O /tmp/twitch.zip \
    https://github.com/lay295/TwitchDownloader/releases/download/1.56.4/TwitchDownloaderCLI-1.56.4-Linux-x64.zip \
    && unzip -o /tmp/twitch.zip -d /tmp/twitch \
    && mv /tmp/twitch/TwitchDownloaderCLI /usr/local/bin/TwitchDownloaderCLI \
    && chmod +x /usr/local/bin/TwitchDownloaderCLI \
    && rm -rf /tmp/twitch /tmp/twitch.zip

WORKDIR /app

COPY requirements.txt .

# GPU-enabled PyTorch build for Whisper (CUDA 12.1 wheels).
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p output

CMD ["python", "-u","main.py"]
