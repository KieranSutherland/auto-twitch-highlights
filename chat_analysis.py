import json
import pandas as pd
from collections import Counter
from config import CHAT_WINDOW_SEC, LAUGH_EMOTES, EXCITED_WORDS

def load_chat(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    rows = []

    for m in data["comments"]:
        msg = m["message"]
        text = msg.get("body", "")

        emotes = []
        for e in msg.get("emoticons", []):
            if "name" in e:
                emotes.append(e["name"])

        rows.append({
            "time": m["content_offset_seconds"],
            "text": text,
            "emotes": emotes
        })

    return pd.DataFrame(rows)

def contains_laugh(row):
    text = row["text"].lower()
    emotes = set(row["emotes"])

    if any(e in emotes for e in LAUGH_EMOTES):
        return True

    if any(word in text for word in ["lol", "lmao", "haha", "rofl"]):
        return True

    return False


def chat_signals(df):
    times = df["time"].astype(int)
    signal = []

    for t in range(int(times.max())):
        window = df[(df["time"] >= t) & (df["time"] < t + CHAT_WINDOW_SEC)]
        if len(window) == 0:
            signal.append(0)
            continue

        laugh_count = sum(
            contains_laugh(row)
            for _, row in window.iterrows()
        )


        caps_ratio = sum(
            row["text"].isupper() for _, row in window.iterrows()
        ) / len(window)

        excited_words = sum(
            any(w in row["text"].lower() for w in EXCITED_WORDS)
            for _, row in window.iterrows()
        )

        score = (
            laugh_count * 2 +
            excited_words * 2 +
            caps_ratio * 3 +
            len(window) / 10
        )

        signal.append(score)

    return signal
