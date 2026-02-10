import config
from chat_analysis import contains_laugh

EXCITED_WORDS = config.EXCITED_WORDS
CHAT_REACTION_MIN_DELAY_SEC = getattr(config, "CHAT_REACTION_MIN_DELAY_SEC", 3.0)
CHAT_REACTION_MAX_DELAY_SEC = getattr(config, "CHAT_REACTION_MAX_DELAY_SEC", 5.0)
CHAT_REACTION_WINDOW_SEC = getattr(config, "CHAT_REACTION_WINDOW_SEC", 12.0)
MAX_HIGHLIGHTS = getattr(config, "MAX_HIGHLIGHTS", 10)
MIN_HIGHLIGHT_SCORE = getattr(config, "MIN_HIGHLIGHT_SCORE", 3.0)
RELATIVE_SCORE_FLOOR = getattr(config, "RELATIVE_SCORE_FLOOR", 0.35)


def _score_chat_window(window, window_duration_sec):
    if len(window) == 0:
        return 0.0

    laugh_count = sum(contains_laugh(row) for _, row in window.iterrows())
    excited_words = sum(
        any(word in row["text"].lower() for word in EXCITED_WORDS)
        for _, row in window.iterrows()
    )
    caps_ratio = sum(
        bool(row["text"]) and row["text"].isupper()
        for _, row in window.iterrows()
    ) / len(window)

    duration = max(window_duration_sec, 1.0)
    laugh_rate = laugh_count / duration
    excited_rate = excited_words / duration
    message_rate = len(window) / duration

    return (
        laugh_rate * 40.0 +
        excited_rate * 25.0 +
        caps_ratio * 4.0 +
        message_rate * 2.0
    )


def _score_audio_window(audio_sig, start_sec, end_sec):
    if not audio_sig:
        return 0.0

    start = max(0, int(start_sec))
    end = min(len(audio_sig), int(end_sec) + 1)
    if end <= start:
        return 0.0

    values = [max(v, 0.0) for v in audio_sig[start:end]]
    if not values:
        return 0.0

    return sum(values) / len(values)


def _delay_candidates(min_delay_sec, max_delay_sec):
    low = min(float(min_delay_sec), float(max_delay_sec))
    high = max(float(min_delay_sec), float(max_delay_sec))

    if abs(high - low) < 0.01:
        return [low]

    mid = (low + high) / 2.0
    return [low, mid, high]


def rank_thoughts_for_highlights(
    thought_segments,
    chat_df,
    audio_sig=None,
    reaction_delay_min_sec=CHAT_REACTION_MIN_DELAY_SEC,
    reaction_delay_max_sec=CHAT_REACTION_MAX_DELAY_SEC,
    reaction_window_sec=CHAT_REACTION_WINDOW_SEC,
):
    scored = []
    delays = _delay_candidates(reaction_delay_min_sec, reaction_delay_max_sec)
    center_delay = sum(delays) / len(delays)

    for thought in thought_segments:
        start = float(thought["start"])
        end = float(thought["end"])
        if end <= start:
            continue

        delay_scores = []
        delay_weights = []
        for delay in delays:
            reaction_start = start + delay
            reaction_end = end + delay + reaction_window_sec

            chat_window = chat_df[
                (chat_df["time"] >= reaction_start) &
                (chat_df["time"] < reaction_end)
            ]

            delay_scores.append(_score_chat_window(chat_window, reaction_end - reaction_start))

            # Weight delays closest to center higher, but still include edges.
            distance = abs(delay - center_delay)
            delay_weights.append(1.0 / (1.0 + distance))

        if delay_scores:
            weighted_avg = sum(
                score * weight for score, weight in zip(delay_scores, delay_weights)
            ) / max(sum(delay_weights), 1e-6)
            chat_score = (0.6 * weighted_avg) + (0.4 * max(delay_scores))
        else:
            chat_score = 0.0

        audio_score = _score_audio_window(audio_sig, start, end)

        scored.append({
            "start": start,
            "end": end,
            "text": thought.get("text", ""),
            "score": chat_score + (0.4 * audio_score),
            "chat_score": chat_score,
            "audio_score": audio_score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _overlaps(candidate, selected):
    for existing in selected:
        if candidate["start"] < existing["end"] and candidate["end"] > existing["start"]:
            return True
    return False


def select_top_highlights(
    scored_thoughts,
    max_highlights=MAX_HIGHLIGHTS,
    min_score=MIN_HIGHLIGHT_SCORE,
    relative_floor=RELATIVE_SCORE_FLOOR,
):
    if not scored_thoughts:
        return []

    top_score = scored_thoughts[0]["score"]
    score_floor = max(min_score, top_score * relative_floor)

    selected = []
    for thought in scored_thoughts:
        if thought["score"] < score_floor:
            continue

        if _overlaps(thought, selected):
            continue

        selected.append(thought)
        if len(selected) >= max_highlights:
            break

    if not selected:
        selected = [scored_thoughts[0]]

    selected.sort(key=lambda x: x["start"])
    return selected


def find_highlights_from_thoughts(
    thought_segments,
    chat_df,
    audio_sig=None,
    reaction_delay_min_sec=CHAT_REACTION_MIN_DELAY_SEC,
    reaction_delay_max_sec=CHAT_REACTION_MAX_DELAY_SEC,
    reaction_window_sec=CHAT_REACTION_WINDOW_SEC,
):
    scored = rank_thoughts_for_highlights(
        thought_segments,
        chat_df,
        audio_sig=audio_sig,
        reaction_delay_min_sec=reaction_delay_min_sec,
        reaction_delay_max_sec=reaction_delay_max_sec,
        reaction_window_sec=reaction_window_sec,
    )
    selected = select_top_highlights(scored)
    return [(item["start"], item["end"]) for item in selected], selected
