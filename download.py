import subprocess
import os
from config import OUTPUT_DIR, VOD_URL
import re
import json

CHAT_BACKGROUND_COLOR = "#00000000"  # Fully transparent background
CHAT_HEIGHT = "600"
CHAT_WIDTH = "450"
CHAT_FONT_SIZE = "26"

def _find_chat_mask_path():
    preferred_names = (
        "chat_mask.mp4",
        "chat-mask.mp4",
        "chat.mask.mp4",
        "chat_mask.webm",
        "chat-mask.webm",
        "chat.mask.webm",
        "chat_mask.mkv",
        "chat-mask.mkv",
        "chat.mask.mkv",
    )
    for name in preferred_names:
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            return path

    # Fallback: pick the newest mask-like video file in output/.
    candidates = []
    for name in os.listdir(OUTPUT_DIR):
        lowered = name.lower()
        if "mask" not in lowered:
            continue
        if not lowered.endswith((".mp4", ".webm", ".mkv", ".mov")):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isfile(path):
            candidates.append((os.path.getmtime(path), path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]

def download_vod():
    if os.path.exists(f"{OUTPUT_DIR}/vod.mp4"):
        print("VOD already exists, skipping download")
        return
    print(f"\nDownloading VOD...\n")
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


def fetch_vod_owner_info():
    vod_id = extract_vod_id()
    try:
        result = subprocess.run([
            "TwitchDownloaderCLI",
            "info",
            "--id", vod_id,
            "--format", "Raw",
        ], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Failed to fetch VOD info via TwitchDownloaderCLI for VOD {vod_id}: {stderr}"
        ) from exc

    raw_output = f"{result.stdout or ''}\n{result.stderr or ''}"
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        owner = (((payload.get("data") or {}).get("video") or {}).get("owner") or {})
        display_name = owner.get("displayName")
        login = owner.get("login")
        if display_name and login:
            return {
                "display_name": display_name,
                "login": login,
            }

    raise RuntimeError(
        f"Could not extract owner displayName/login from TwitchDownloaderCLI info output for VOD {vod_id}"
    )

def download_chat():
    if os.path.exists(f"{OUTPUT_DIR}/chat.json"):
        print("Chat already exists, skipping download")
        return
    print(f"\nDownloading chat...\n")
    vod_id = extract_vod_id()
    subprocess.run([
        "TwitchDownloaderCLI",
        "chatdownload",
        "--id", vod_id,
        "--output", f"{OUTPUT_DIR}/chat.json"
    ])

def render_chat():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chat_path = f"{OUTPUT_DIR}/chat.mp4"
    chat_json_path = f"{OUTPUT_DIR}/chat.json"
    if not os.path.exists(chat_json_path):
        raise FileNotFoundError(
            f"Chat JSON not found at {chat_json_path}. Run chat download first."
        )
    existing_mask = _find_chat_mask_path()
    if os.path.exists(chat_path) and existing_mask:
        print("Chat render and mask already exist, skipping generation")
        return existing_mask

    print(f"\nRendering chat...\n")
    mask_requested = True
    try:
        subprocess.run([
            "TwitchDownloaderCLI",
            "chatrender",
            "--generate-mask",
            "--background-color", CHAT_BACKGROUND_COLOR,
            "--chat-height", CHAT_HEIGHT,
            "--chat-width", CHAT_WIDTH,
            "--font-size", CHAT_FONT_SIZE,
            "--input", chat_json_path,
            "--output", chat_path,
        ], check=True)
    except subprocess.CalledProcessError:
        # Fallback for older TwitchDownloaderCLI builds that don't support --generate-mask.
        mask_requested = False
        print("Mask generation failed; retrying chat render without --generate-mask.")
        subprocess.run([
            "TwitchDownloaderCLI",
            "chatrender",
            "--background-color", CHAT_BACKGROUND_COLOR,
            "--chat-height", CHAT_HEIGHT,
            "--chat-width", CHAT_WIDTH,
            "--font-size", CHAT_FONT_SIZE,
            "--input", chat_json_path,
            "--output", chat_path,
        ], check=True)

    if not os.path.exists(chat_path):
        raise RuntimeError(
            f"Chat render failed to produce {chat_path}; cannot continue vertical output mode."
        )

    mask_path = _find_chat_mask_path()
    if mask_path:
        print(f"Detected chat mask: {mask_path}")
    elif mask_requested:
        print("Warning: --generate-mask completed but no mask file was found.")
    else:
        print("Warning: chat mask not found; using fallback chat overlay.")
    return mask_path
