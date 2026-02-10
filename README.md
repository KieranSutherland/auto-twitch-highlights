# Auto Twitch Highlights
Automatically generate highlight clips from a twitch video.

## Build the docker image (GPU-ready)
`docker build -t auto-twitch-highlights .`

## Run with GPU
Requires NVIDIA Container Toolkit and a CUDA-capable GPU.

`docker run --rm --gpus all -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights`

`docker volume rm twitch-highlights-volume && docker volume create twitch-highlights-volume && docker run --rm --gpus all -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights`

`export RUN_WITH_SAVED="docker run --rm --gpus all -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights"`
`export RUN_FULL="docker volume rm twitch-highlights-volume && docker volume create twitch-highlights-volume && docker run --rm --gpus all -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights"`

## Optional Whisper settings
`WHISPER_MODEL` defaults to `base`.
`WHISPER_DEVICE` defaults to `cuda` in Docker and automatically falls back to CPU if CUDA is unavailable.

## Output layout
Each run writes into a VOD-specific folder:
- `output/<vod_id>/` where `<vod_id>` is the number after `videos/` in `VOD_URL`.

Clips are rendered as phone format (`1080x1920`):
- Top (full width): auto-detected facecam panel from `output/<vod_id>/vod.mp4`.
- Middle/background: gameplay, center-cropped (at least 20% removed from left and right), then additionally cropped as needed so width matches facecam/chat.
- Bottom-left (overlapping gameplay): rendered chat from `output/<vod_id>/chat.mp4`.
  - Chat rendering now uses transparent background and requests `--generate-mask` from TwitchDownloaderCLI.
  - If a mask file is found, it is applied as alpha for a true transparent overlay.

Facecam and chat are drawn above gameplay so they remain visible and prioritized.

Per run outputs include:
- Short-form clips: `output/<vod_id>/clip_00.mp4`, `clip_01.mp4`, etc.
- Long-form compilation (1080p, chat pinned bottom-left): `output/<vod_id>/highlights_longform_1080p.mp4`.

If `output/<vod_id>/chat.mp4` is missing, the pipeline falls back to standard clip rendering.
