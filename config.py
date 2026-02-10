import os
import re

VOD_URL = os.getenv("VOD_URL")
if not VOD_URL:
    raise RuntimeError("VOD_URL environment variable not set")

vod_id_match = re.search(r"videos/(\d+)", VOD_URL)
if not vod_id_match:
    raise RuntimeError("Could not extract VOD ID from VOD_URL")

VOD_ID = vod_id_match.group(1)
OUTPUT_ROOT_DIR = "output"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, VOD_ID)

# Time settings
CHAT_WINDOW_SEC = 5
AUDIO_WINDOW_SEC = 3
HIGHLIGHT_THRESHOLD = 6
CLIP_PADDING_BEFORE = 8
CLIP_PADDING_AFTER = 6

# Chat signals
LAUGH_EMOTES = {"Kappa", "LUL", "KEKW", "😂", "🤣", "OMEGALUL"}
EXCITED_WORDS = {"clip", "no way", "wtf", "lol", "lmao"}
