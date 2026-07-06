# PUREAM MuseTalk Lip Sync

HTTP service for MuseTalk lip-sync deployment on FunctionAI.

## API

```http
POST /
content-type: application/json
```

```json
{
  "video_url": "https://example.com/input.mp4",
  "audio_url": "https://example.com/voice.wav"
}
```

The service downloads the video and audio, trims the video to the audio duration, runs MuseTalk, muxes the target audio, uploads the result to OSS, and returns the generated video URL.

## Required runtime environment variables

```text
OSS_AK
OSS_SK
OSS_BUCKET
OSS_REGION=oss-cn-hangzhou
OSS_PREFIX=puream/lipsync
OSS_PUBLIC_BASE_URL=
PUREAM_FC_TOKEN=
```

`PUREAM_FC_TOKEN` is optional. If set, requests must include `x-puream-fc-token`.

## Image

GitHub Actions publishes:

```text
ghcr.io/gxy19941008/puream-lipsync-musetalk:latest
```

Use this image in FunctionAI custom environment.

Recommended FunctionAI settings:

```text
Port: 9000
Timeout: 1800
Instance concurrency: match GPU card count, usually 1 for 1 card, 4 for 4 cards
Minimum instances: 0
Public access: enabled
Auth: disabled during testing
```
