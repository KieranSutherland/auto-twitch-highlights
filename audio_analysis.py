import config
import ffmpeg
import librosa
import os
import re
import whisper

try:
    import torch
except Exception:  # pragma: no cover - fallback for minimal installs
    torch = None

OUTPUT_DIR = config.OUTPUT_DIR
THOUGHT_GAP_SEC = getattr(config, "THOUGHT_GAP_SEC", 1.5)
THOUGHT_MIN_SEC = getattr(config, "THOUGHT_MIN_SEC", 2.5)
THOUGHT_MAX_SEC = getattr(config, "THOUGHT_MAX_SEC", 180.0)
LAST_WORD_TAIL_SEC = getattr(config, "LAST_WORD_TAIL_SEC", 0.28)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE_OVERRIDE = os.getenv("WHISPER_DEVICE", "").strip().lower()

TERMINAL_PUNCTUATION = (".", "!", "?", "...")
CONTINUATION_WORDS = {
    "and", "but", "or", "so", "because", "then", "if", "when",
    "that", "which", "who", "to", "for", "with", "of", "like",
}
TOPIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "for", "of",
    "in", "on", "at", "it", "its", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "i", "you",
    "he", "she", "they", "we", "me", "my", "your", "our", "their",
}


def extract_audio():
    if os.path.exists(f"{OUTPUT_DIR}/audio.wav"):
        print("Audio already exists, skipping extraction")
        return
    print(f"\nExtracting audio...\n")
    (
        ffmpeg
        .input(f"{OUTPUT_DIR}/vod.mp4")
        .output(f"{OUTPUT_DIR}/audio.wav", vn=None)
        .overwrite_output()
        .run(quiet=True)
    )


def audio_signals(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    std = rms.std()
    if std == 0:
        return [0.0 for _ in rms]

    rms_norm = (rms - rms.mean()) / std
    return rms_norm.tolist()


def _resolve_whisper_device():
    if WHISPER_DEVICE_OVERRIDE in {"cpu", "cuda"}:
        if WHISPER_DEVICE_OVERRIDE == "cuda":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            print("\nWHISPER_DEVICE=cuda requested but CUDA not available; falling back to CPU.\n")
            return "cpu"

        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def analyze_speech_boundaries(audio_path, model_name=None):
    """
    Use Whisper to transcribe audio and return speech segments with timestamps.
    """
    model_name = model_name or WHISPER_MODEL
    device = _resolve_whisper_device()
    use_fp16 = device == "cuda"

    print(f"\nLoading Whisper model ({model_name})...\n")
    print(f"\nUsing Whisper device: {device}\n")
    model = whisper.load_model(model_name, device=device)

    print(f"\nTranscribing audio...\n")
    result = model.transcribe(
        audio_path,
        word_timestamps=False,
        verbose=False,
        fp16=use_fp16
    )

    segments = []
    for segment in result.get("segments", []):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        text = segment.get("text", "").strip()

        if end <= start or not text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": text,
            "no_speech_prob": float(segment.get("no_speech_prob", 0)),
        })

    return segments


def _is_terminal_text(text):
    text = (text or "").strip()
    if not text:
        return False

    tail = text.rstrip(')"\' ]}')
    return tail.endswith(TERMINAL_PUNCTUATION)


def _starts_like_continuation(text):
    text = (text or "").strip()
    if not text:
        return False

    first_char = text[0]
    if first_char.islower():
        return True

    first_word = text.split()[0].strip('("[]{}').lower()
    return first_word in CONTINUATION_WORDS


def _tokenize_topic_words(text):
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    return [
        w for w in words
        if len(w) >= 3 and w not in TOPIC_STOPWORDS
    ]


def _token_overlap_score(left_text, right_text):
    left = set(_tokenize_topic_words(left_text))
    right = set(_tokenize_topic_words(right_text))
    if not left or not right:
        return 0.0

    return len(left & right) / float(min(len(left), len(right)))


def _clip_tail_text(speech_segments, right_idx, window_segments=3):
    start_idx = max(0, right_idx - window_segments + 1)
    parts = [speech_segments[i].get("text", "") for i in range(start_idx, right_idx + 1)]
    return " ".join(parts).strip()


def _clip_head_text(speech_segments, left_idx, window_segments=3):
    end_idx = min(len(speech_segments), left_idx + window_segments)
    parts = [speech_segments[i].get("text", "") for i in range(left_idx, end_idx)]
    return " ".join(parts).strip()


def _bridge_stats(speech_segments, prev_right_idx, next_left_idx, clip_gap_sec):
    if next_left_idx <= (prev_right_idx + 1):
        return 0.0, ""

    bridge_segments = speech_segments[prev_right_idx + 1:next_left_idx]
    bridge_speech_sec = sum(
        float(seg["end"]) - float(seg["start"])
        for seg in bridge_segments
    )
    bridge_density = bridge_speech_sec / max(clip_gap_sec, 0.001)
    bridge_text = " ".join(seg.get("text", "") for seg in bridge_segments).strip()
    return bridge_density, bridge_text


def _clips_are_related(
    prev_clip,
    next_clip,
    speech_segments,
    merge_gap_sec,
    related_gap_sec,
    close_gap_sec,
    sentiment_cluster_gap_sec,
):
    gap = next_clip["start"] - prev_clip["end"]
    if gap <= merge_gap_sec:
        return True

    # Very short pauses are typically one continuous sentiment.
    if gap <= 5.0:
        return True

    if gap > related_gap_sec:
        return False

    prev_tail = _clip_tail_text(speech_segments, prev_clip["right_idx"])
    next_head = _clip_head_text(speech_segments, next_clip["left_idx"])

    continuation = (
        not _is_terminal_text(prev_tail) or
        _starts_like_continuation(next_head)
    )
    tail_head_overlap = _token_overlap_score(prev_tail, next_head)

    bridge_density, bridge_text = _bridge_stats(
        speech_segments,
        prev_clip["right_idx"],
        next_clip["left_idx"],
        gap,
    )
    bridge_overlap = max(
        _token_overlap_score(prev_tail, bridge_text),
        _token_overlap_score(bridge_text, next_head),
    ) if bridge_text else 0.0

    # Close-in-time clips should merge unless there is a clear topic break.
    if gap <= close_gap_sec:
        hard_break = (
            _is_terminal_text(prev_tail) and
            not _starts_like_continuation(next_head) and
            tail_head_overlap < 0.08 and
            bridge_overlap < 0.08
        )
        if not hard_break:
            return True

    # Medium gaps can still be the same sentiment if speech bridge is strong.
    if gap <= sentiment_cluster_gap_sec:
        if bridge_density >= 0.5:
            return True
        if bridge_density >= 0.25 and (
            continuation or
            tail_head_overlap >= 0.06 or
            bridge_overlap >= 0.08
        ):
            return True

    score = 0.0
    if continuation:
        score += 1.2
    if tail_head_overlap >= 0.12:
        score += 0.8
    if tail_head_overlap >= 0.25:
        score += 0.6
    if bridge_density >= 0.35:
        score += 0.7
    if bridge_overlap >= 0.12:
        score += 0.6
    if gap <= 8.0:
        score += 0.4

    if gap <= sentiment_cluster_gap_sec and score >= 1.2:
        return True

    return score >= 1.6


def build_thought_segments(
    speech_segments,
    max_gap_sec=THOUGHT_GAP_SEC,
    min_duration_sec=THOUGHT_MIN_SEC,
    max_duration_sec=THOUGHT_MAX_SEC,
):
    """
    Merge Whisper sentence segments into thought-level chunks.
    """
    thoughts = []
    current = None

    def flush():
        nonlocal current
        if not current:
            return

        duration = current["end"] - current["start"]
        if duration >= min_duration_sec and current["text"].strip():
            thoughts.append(current)

        current = None

    for segment in speech_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_text = segment.get("text", "").strip()

        if seg_end <= seg_start or not seg_text:
            continue

        if current is None:
            current = {
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "segment_count": 1,
            }
            continue

        gap = seg_start - current["end"]
        projected_duration = seg_end - current["start"]

        current_open_ended = not _is_terminal_text(current["text"])
        next_looks_continuation = _starts_like_continuation(seg_text)

        should_merge = False
        if gap <= max_gap_sec and projected_duration <= max_duration_sec:
            should_merge = True
        elif (
            gap <= (max_gap_sec + 1.5)
            and projected_duration <= max_duration_sec
            and (current_open_ended or next_looks_continuation)
        ):
            # Give extra leeway when sentence punctuation suggests the thought is not finished.
            should_merge = True

        if should_merge:
            current["end"] = seg_end
            current["text"] = f"{current['text']} {seg_text}".strip()
            current["segment_count"] += 1
            continue

        flush()
        current = {
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "segment_count": 1,
        }

    flush()

    if thoughts:
        return thoughts

    # Fallback for streams where Whisper produces many short isolated segments.
    fallback = []
    for segment in speech_segments:
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", seg_start))
        seg_text = segment.get("text", "").strip()
        if seg_end > seg_start and seg_text:
            fallback.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "segment_count": 1,
            })

    return fallback


def _find_covering_segment_range(speech_segments, start, end, allow_nearest_fallback=True):
    first_idx = None
    last_idx = None

    for idx, seg in enumerate(speech_segments):
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        overlaps = seg_end > start and seg_start < end
        if overlaps:
            if first_idx is None:
                first_idx = idx
            last_idx = idx

    if first_idx is not None:
        return first_idx, last_idx

    if not allow_nearest_fallback:
        return None, None

    if not speech_segments:
        return None, None

    best_idx = min(
        range(len(speech_segments)),
        key=lambda i: abs(float(speech_segments[i]["start"]) - start),
    )
    return best_idx, best_idx


def expand_highlights_to_complete_thoughts(
    highlights,
    speech_segments,
    max_context_sec=45.0,
    continuity_gap_sec=4.0,
    min_clip_sec=10.0,
    max_clip_sec=180.0,
):
    """
    Expand selected highlight ranges to avoid starting/ending in the middle of a thought.
    """
    if not highlights or not speech_segments:
        return highlights

    sorted_segments = sorted(speech_segments, key=lambda s: float(s["start"]))
    expanded = []

    for start, end in highlights:
        if end <= start:
            continue

        first_idx, last_idx = _find_covering_segment_range(sorted_segments, start, end)
        if first_idx is None:
            expanded.append((start, end))
            continue

        left_idx = first_idx
        right_idx = last_idx
        original_start = float(start)
        original_end = float(end)
        clip_start = float(sorted_segments[left_idx]["start"])
        clip_end = float(sorted_segments[right_idx]["end"])

        while left_idx > 0:
            prev_seg = sorted_segments[left_idx - 1]
            cur_seg = sorted_segments[left_idx]
            gap = float(cur_seg["start"]) - float(prev_seg["end"])
            proposed_start = float(prev_seg["start"])
            proposed_duration = clip_end - proposed_start
            within_context = (original_start - proposed_start) <= max_context_sec

            if gap > continuity_gap_sec or proposed_duration > max_clip_sec or not within_context:
                break

            tight_gap = gap <= 0.5
            should_extend = (
                tight_gap or
                _starts_like_continuation(cur_seg.get("text", "")) or
                not _is_terminal_text(prev_seg.get("text", "")) or
                (clip_end - clip_start) < min_clip_sec
            )
            if not should_extend:
                break

            left_idx -= 1
            clip_start = proposed_start

        while right_idx < (len(sorted_segments) - 1):
            cur_seg = sorted_segments[right_idx]
            next_seg = sorted_segments[right_idx + 1]
            gap = float(next_seg["start"]) - float(cur_seg["end"])
            proposed_end = float(next_seg["end"])
            proposed_duration = proposed_end - clip_start
            within_context = (proposed_end - original_end) <= max_context_sec

            if gap > continuity_gap_sec or proposed_duration > max_clip_sec or not within_context:
                break

            tight_gap = gap <= 0.5
            should_extend = (
                tight_gap or
                not _is_terminal_text(cur_seg.get("text", "")) or
                _starts_like_continuation(next_seg.get("text", "")) or
                (clip_end - clip_start) < min_clip_sec
            )
            if not should_extend:
                break

            right_idx += 1
            clip_end = proposed_end

        duration = clip_end - clip_start
        if duration >= min_clip_sec and duration <= max_clip_sec:
            expanded.append((clip_start, clip_end))

    # Merge accidental overlaps introduced by context expansion.
    expanded.sort(key=lambda x: x[0])
    merged = []
    for start, end in expanded:
        if not merged:
            merged.append([start, end])
            continue

        prev_start, prev_end = merged[-1]
        merged_end = max(prev_end, end)
        merged_duration = merged_end - prev_start

        if start <= prev_end and merged_duration <= max_clip_sec:
            merged[-1][1] = merged_end
        else:
            merged.append([start, end])

    return [
        (s, e)
        for s, e in merged
        if (e - s) >= min_clip_sec and (e - s) <= max_clip_sec
    ]


def finalize_highlights_to_speech(
    highlights,
    speech_segments,
    merge_gap_sec=1.2,
    close_gap_sec=6.0,
    related_gap_sec=35.0,
    sentiment_cluster_gap_sec=30.0,
    min_clip_sec=10.0,
    max_clip_sec=180.0,
):
    """
    Snap highlights to speech segment boundaries, merge near-adjacent clips from the same sentiment,
    and enforce strict non-overlap + duration constraints.
    """
    if not highlights or not speech_segments:
        return []

    sorted_segments = sorted(speech_segments, key=lambda s: float(s["start"]))
    snapped = []

    for start, end in highlights:
        if end <= start:
            continue

        left_idx, right_idx = _find_covering_segment_range(
            sorted_segments, start, end, allow_nearest_fallback=False
        )
        if left_idx is None:
            continue

        clip_start = float(sorted_segments[left_idx]["start"])
        clip_end = float(sorted_segments[right_idx]["end"])

        # If a snapped clip is slightly too short, extend into immediately adjacent speech.
        while (clip_end - clip_start) < min_clip_sec:
            left_gap = float("inf")
            right_gap = float("inf")

            if left_idx > 0:
                prev_seg = sorted_segments[left_idx - 1]
                cur_left_seg = sorted_segments[left_idx]
                left_gap = float(cur_left_seg["start"]) - float(prev_seg["end"])

            if right_idx < (len(sorted_segments) - 1):
                cur_right_seg = sorted_segments[right_idx]
                next_seg = sorted_segments[right_idx + 1]
                right_gap = float(next_seg["start"]) - float(cur_right_seg["end"])

            extended = False

            if left_gap <= right_gap and left_gap <= merge_gap_sec and left_idx > 0:
                proposed_start = float(sorted_segments[left_idx - 1]["start"])
                if (clip_end - proposed_start) <= max_clip_sec:
                    left_idx -= 1
                    clip_start = proposed_start
                    extended = True

            if (
                not extended and
                right_gap <= merge_gap_sec and
                right_idx < (len(sorted_segments) - 1)
            ):
                proposed_end = float(sorted_segments[right_idx + 1]["end"])
                if (proposed_end - clip_start) <= max_clip_sec:
                    right_idx += 1
                    clip_end = proposed_end
                    extended = True

            if not extended:
                break

        duration = clip_end - clip_start
        if duration >= min_clip_sec and duration <= max_clip_sec:
            snapped.append({
                "start": clip_start,
                "end": clip_end,
                "left_idx": left_idx,
                "right_idx": right_idx,
            })

    if not snapped:
        return []

    snapped.sort(key=lambda x: x["start"])
    merged = []

    for candidate in snapped:
        start = candidate["start"]
        end = candidate["end"]

        if not merged:
            merged.append(dict(candidate))
            continue

        prev = merged[-1]
        gap = start - prev["end"]
        proposed_end = max(prev["end"], end)
        proposed_duration = proposed_end - prev["start"]

        if (
            proposed_duration <= max_clip_sec and
            _clips_are_related(
                prev,
                candidate,
                sorted_segments,
                merge_gap_sec,
                related_gap_sec,
                close_gap_sec,
                sentiment_cluster_gap_sec,
            )
        ):
            # Conjoin clips when transcript continuity indicates one ongoing sentiment.
            prev["end"] = proposed_end
            prev["left_idx"] = min(prev["left_idx"], candidate["left_idx"])
            prev["right_idx"] = max(prev["right_idx"], candidate["right_idx"])
            continue

        if start < prev["end"]:
            # If overlapping but cannot merge due max length, drop later clip to keep timeline clean.
            continue

        merged.append(dict(candidate))

    return [
        (item["start"], item["end"])
        for item in merged
        if (item["end"] - item["start"]) >= min_clip_sec and (item["end"] - item["start"]) <= max_clip_sec
    ]


def build_clip_manifests(
    highlights,
    speech_segments,
    last_word_tail_sec=LAST_WORD_TAIL_SEC,
    min_clip_sec=10.0,
    max_clip_sec=180.0,
):
    """
    Add a tiny end-tail to reduce last-word cutoffs.
    """
    if not highlights:
        return []

    ordered = sorted((float(s), float(e)) for s, e in highlights if e > s)
    manifests = []

    for idx, (start, end) in enumerate(ordered):
        next_start = ordered[idx + 1][0] if idx + 1 < len(ordered) else None
        tuned_end = min(end + float(last_word_tail_sec), start + max_clip_sec)

        # Keep non-overlap after adding the tail; leave a tiny safety margin.
        if next_start is not None and tuned_end >= next_start:
            tuned_end = max(start, next_start - 0.01)

        duration = tuned_end - start
        if duration < min_clip_sec or duration > max_clip_sec:
            continue

        manifests.append({
            "start": start,
            "end": tuned_end,
        })

    return manifests
