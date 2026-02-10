import os

VOD_URL = os.getenv("VOD_URL")
if not VOD_URL:
    raise RuntimeError("VOD_URL environment variable not set")

OUTPUT_DIR = "output"

# Time settings
CHAT_WINDOW_SEC = 5
AUDIO_WINDOW_SEC = 3
HIGHLIGHT_THRESHOLD = 6
CLIP_PADDING_BEFORE = 8
CLIP_PADDING_AFTER = 6

# Chat signals
LAUGH_EMOTES = {"Kappa", "LUL", "KEKW", "😂", "🤣", "OMEGALUL"}
EXCITED_WORDS = {"clip", "no way", "wtf", "lol", "lmao"}
