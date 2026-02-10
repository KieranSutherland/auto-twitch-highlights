FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    ca-certificates \
    libicu-dev \
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
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output

CMD ["python", "-u","main.py"]
