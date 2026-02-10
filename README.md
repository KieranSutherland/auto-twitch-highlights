# Auto Twitch Highlights
Automatically generate highlight clips from a twitch video.

## Build the docker image
`docker build -t auto-twitch-highlights .`

## Example command to run the docker container
`docker run --rm -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights`
`docker volume rm twitch-highlights-volume && docker volume create twitch-highlights-volume && docker run --rm -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights`

`export RUN_WITH_SAVED="docker run --rm -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights"`
`export RUN_FULL="docker volume rm twitch-highlights-volume && docker volume create twitch-highlights-volume && docker run --rm -e VOD_URL="https://www.twitch.tv/videos/2676476066" -v twitch-highlights-volume:/app/output auto-twitch-highlights"`