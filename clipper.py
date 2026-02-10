import ffmpeg
from config import CLIP_PADDING_BEFORE, CLIP_PADDING_AFTER

def cut_clips(video_path, highlights, out_dir):
    for i, (start, end) in enumerate(highlights):
        s = max(0, start - CLIP_PADDING_BEFORE)
        e = end + CLIP_PADDING_AFTER
        print(f"Cutting clip {i}: {s}s to {e}s")

        (
            ffmpeg
            .input(video_path, ss=s, to=e)
            .output(f"{out_dir}/clip_{i}.mp4", c="copy")
            .run(overwrite_output=True, quiet=True)
        )
