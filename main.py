from download import download_vod, download_chat, render_chat, fetch_vod_owner_info
from chat_analysis import load_chat
from audio_analysis import (
    analyze_speech_boundaries,
    build_thought_segments,
    build_clip_manifests,
    expand_highlights_to_complete_thoughts,
    finalize_highlights_to_speech,
    audio_signals,
    extract_audio,
)
from highlight_scoring import find_highlights_from_thoughts
from clipper import cut_clips, render_longform_with_chat
from config import OUTPUT_DIR, VOD_ID


def _format_hhmmss(seconds):
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_longform_timestamps(clips):
    lines = []
    timeline_cursor = 0.0

    for item in clips:
        if isinstance(item, dict):
            start = float(item["start"])
            end = float(item["end"])
        else:
            start, end = item
            start = float(start)
            end = float(end)

        if end <= start:
            continue

        lines.append(_format_hhmmss(timeline_cursor))
        timeline_cursor += (end - start)

    return "\n".join(lines)


def _write_longform_description(output_dir, display_name, login, clips):
    timestamps = _build_longform_timestamps(clips)
    if not timestamps:
        timestamps = "00:00:00"

    content = (
        f"This is an unofficial channel, I am not {display_name}. "
        "A link to their channel can be found below.\n\n"
        f"{display_name}'s Twitch: https://www.twitch.tv/{login}\n\n"
        f"{timestamps}\n"
    )
    out_path = f"{output_dir}/highlights_longform_description.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def main():
    print(f"\nRun output folder: {OUTPUT_DIR}\n")
    download_vod()
    download_chat()
    chat_mask_path = render_chat()
    extract_audio()

    chat = load_chat(f"{OUTPUT_DIR}/chat.json")
    if chat.empty:
        raise RuntimeError("Chat file loaded but contains no messages")

    audio_sig = audio_signals(f"{OUTPUT_DIR}/audio.wav")

    print(f"\nAnalyzing speech boundaries with Whisper...\n")
    speech_segments = analyze_speech_boundaries(f"{OUTPUT_DIR}/audio.wav")

    print(f"\nBuilding thought-level segments...\n")
    thoughts = build_thought_segments(speech_segments)
    if not thoughts:
        raise RuntimeError("Whisper completed but no thought segments were produced")

    print(f"\nRanking thoughts by chat reaction...\n")
    highlights, ranked = find_highlights_from_thoughts(
        thoughts,
        chat,
        audio_sig=audio_sig,
        reaction_delay_min_sec=3.0,
        reaction_delay_max_sec=5.0,
    )

    if not highlights:
        raise RuntimeError("No highlights were selected after ranking thought segments")

    print(f"\nExpanding highlights to full sentence/thought boundaries...\n")
    original_count = len(highlights)
    highlights = expand_highlights_to_complete_thoughts(
        highlights,
        speech_segments,
        max_context_sec=45.0,
        continuity_gap_sec=4.0,
        min_clip_sec=10.0,
        max_clip_sec=180.0,
    )
    dropped = original_count - len(highlights)
    if dropped > 0:
        print(
            f"  Dropped {dropped} clip candidate(s) that could not be expanded "
            f"to a complete thought inside 10s-180s."
        )

    print(f"\nFinalizing clips to spoken-content boundaries...\n")
    pre_finalize = len(highlights)
    highlights = finalize_highlights_to_speech(
        highlights,
        speech_segments,
        merge_gap_sec=1.2,
        close_gap_sec=6.0,
        related_gap_sec=35.0,
        sentiment_cluster_gap_sec=30.0,
        min_clip_sec=10.0,
        max_clip_sec=180.0,
    )
    if pre_finalize != len(highlights):
        print(
            f"  Consolidated {pre_finalize} candidate(s) into {len(highlights)} "
            f"non-overlapping clip(s) after speech-boundary snapping."
        )

    if not highlights:
        raise RuntimeError(
            "No highlights remain after enforcing full-thought boundaries and 10s-180s duration rules"
        )

    print(f"\nApplying end-tail tuning...\n")
    clip_manifests = build_clip_manifests(
        highlights,
        speech_segments,
        last_word_tail_sec=0.28,
        min_clip_sec=10.0,
        max_clip_sec=180.0,
    )
    if not clip_manifests:
        raise RuntimeError("No clips remain after end-tail tuning")

    print(f"\nSelected {len(clip_manifests)} high-signal thought clips")
    top_ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    for i, clip in enumerate(top_ranked[:3]):
        print(
            f"  Top {i + 1}: {clip['start']:.1f}s - {clip['end']:.1f}s "
            f"(score={clip['score']:.2f}, chat={clip['chat_score']:.2f})"
        )

    for i, clip in enumerate(clip_manifests[:5]):
        print(f"  Final {i + 1}: {clip['start']:.2f}s - {clip['end']:.2f}s")
    
    print(f"\nCutting into clips...\n")
    clip_count = cut_clips(
        f"{OUTPUT_DIR}/vod.mp4",
        clip_manifests,
        OUTPUT_DIR,
        chat_path=f"{OUTPUT_DIR}/chat.mp4",
        chat_mask_path=chat_mask_path,
    )

    print(f"\nGenerated {clip_count} highlight clips")

    print(f"\nFetching streamer info for long-form description...\n")
    owner = fetch_vod_owner_info()
    description_path = _write_longform_description(
        OUTPUT_DIR,
        owner["display_name"],
        owner["login"],
        clip_manifests,
    )
    print(f"Wrote long-form description: {description_path}")

    longform_out = f"{OUTPUT_DIR}/highlights_longform_1080p.mp4"
    print(f"\nBuilding long-form 1080p compilation...\n")
    longform_ok = render_longform_with_chat(
        video_path=f"{OUTPUT_DIR}/vod.mp4",
        highlights=clip_manifests,
        out_path=longform_out,
        chat_path=f"{OUTPUT_DIR}/chat.mp4",
        chat_mask_path=chat_mask_path,
    )
    if longform_ok:
        print(f"Built long-form file: {longform_out}")
        print(f"Completed run for VOD {VOD_ID}")
    else:
        print("Skipped long-form file (no valid clips).")

if __name__ == "__main__":
    main()
