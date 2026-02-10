import subprocess
import os
from config import OUTPUT_DIR, VOD_URL
import re

CHAT_BACKGROUND_COLOR = "#1111114D" # 30% opacity black

def download_vod():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subprocess.run([
        "yt-dlp",
        "--output", f"{OUTPUT_DIR}/vod.mp4",
        VOD_URL
    ])

def extract_vod_id():
    match = re.search(r"videos/(\d+)", VOD_URL)
    if not match:
        raise RuntimeError("Could not extract VOD ID from URL")
    return match.group(1)

def download_chat():
    vod_id = extract_vod_id()
    subprocess.run([
        "TwitchDownloaderCLI",
        "chatdownload",
        "--id", vod_id,
        "--output", f"{OUTPUT_DIR}/chat.json"
    ])

def render_chat():
    subprocess.run([
        "TwitchDownloaderCLI",
        "chatrender",
        "--background-color", CHAT_BACKGROUND_COLOR,
        "--input", f"{OUTPUT_DIR}/chat.json",
        "--output", f"{OUTPUT_DIR}/chat.mp4"
    ])