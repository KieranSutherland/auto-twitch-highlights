from config import HIGHLIGHT_THRESHOLD

def find_highlights(chat_sig, audio_sig):
    highlights = []
    active = False
    start = 0

    for end in range(min(len(chat_sig), len(audio_sig))):
        score = chat_sig[end] + max(audio_sig[end], 0)

        if score > HIGHLIGHT_THRESHOLD and not active:
            start = end
            active = True

        elif score <= HIGHLIGHT_THRESHOLD and active:
            # Check if this range overlaps with any existing highlight
            overlaps = any(
                not (end <= hl_start or start >= hl_end)
                for hl_start, hl_end in highlights
            )
            
            if not overlaps:
                highlights.append((start, end))
            active = False

    return highlights