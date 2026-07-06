import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import oss2
import requests
from flask import Flask, jsonify, request

from musetalk_runner import MuseTalkRunner

app = Flask(__name__)
runner = MuseTalkRunner()


def require_token():
    expected = os.environ.get("PUREAM_FC_TOKEN", "").strip()
    if not expected:
        return
    actual = request.headers.get("x-puream-fc-token", "").strip()
    if actual != expected:
        raise PermissionError("unauthorized")


def run_json(cmd):
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return json.loads(output)


def duration_seconds(path):
    data = run_json([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path,
    ])
    return float(data["format"]["duration"])


def video_info(path):
    data = run_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path,
    ])
    stream = data["streams"][0]
    rate = stream.get("r_frame_rate", "0/1")
    num, den = [float(x) for x in rate.split("/", 1)]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": num / den if den else 0,
    }


def safe_suffix(url, fallback):
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix and len(suffix) <= 8 else fallback


def download(url, path):
    with requests.get(url, stream=True, timeout=(20, 600)) as res:
        res.raise_for_status()
        with open(path, "wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return os.path.getsize(path)


def trim_video_to_audio(video_path, audio_path, output_path):
    audio_duration = duration_seconds(audio_path)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-t", f"{audio_duration:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        output_path,
    ], check=True)
    return audio_duration


def mux_audio(video_path, audio_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
    ], check=True)


def oss_bucket():
    auth = oss2.Auth(os.environ["OSS_AK"], os.environ["OSS_SK"])
    endpoint = f"https://{os.environ.get('OSS_REGION', 'oss-cn-hangzhou')}.aliyuncs.com"
    return oss2.Bucket(auth, endpoint, os.environ["OSS_BUCKET"])


def upload(path, task_id):
    prefix = os.environ.get("OSS_PREFIX", "puream/lipsync").strip("/")
    key = f"{prefix}/{time.strftime('%Y%m%d')}/{task_id}.mp4"
    bucket = oss_bucket()
    content_type = mimetypes.guess_type(path)[0] or "video/mp4"
    bucket.put_object_from_file(key, path, headers={"Content-Type": content_type})
    public_base = os.environ.get("OSS_PUBLIC_BASE_URL", "").rstrip("/")
    if public_base:
        url = f"{public_base}/{key}"
    else:
        region = os.environ.get("OSS_REGION", "oss-cn-hangzhou")
        url = f"https://{os.environ['OSS_BUCKET']}.{region}.aliyuncs.com/{key}"
    return key, url


@app.get("/health")
def health():
    return jsonify({"ok": True, "model": "MuseTalk", "gpu": os.path.exists("/dev/nvidia0")})


@app.post("/")
def lipsync():
    started = time.time()
    task_id = ""
    work_dir = None
    try:
        require_token()
        data = request.get_json(force=True)
        video_url = str(data.get("video_url") or "").strip()
        audio_url = str(data.get("audio_url") or "").strip()
        task_id = str(data.get("task_id") or uuid.uuid4().hex)
        if not video_url or not audio_url:
            return jsonify({"ok": False, "message": "video_url and audio_url are required"}), 400

        work_dir = Path(os.environ.get("WORK_DIR", "/tmp/puream-lipsync")) / task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(work_dir / f"input{safe_suffix(video_url, '.mp4')}")
        audio_path = str(work_dir / f"audio{safe_suffix(audio_url, '.wav')}")
        video_size = download(video_url, video_path)
        audio_size = download(audio_url, audio_path)
        source_duration = duration_seconds(video_path)
        audio_duration = duration_seconds(audio_path)
        if source_duration + 0.05 < audio_duration:
            return jsonify({
                "ok": False,
                "message": "video duration must be longer than or equal to audio duration",
                "original_duration_seconds": source_duration,
                "audio_duration_seconds": audio_duration,
            }), 400

        trimmed = str(work_dir / "video_trimmed.mp4")
        trim_video_to_audio(video_path, audio_path, trimmed)
        raw_output = runner.run(trimmed, audio_path, task_id)
        final_output = str(work_dir / "final.mp4")
        mux_audio(raw_output, audio_path, final_output)
        info = video_info(final_output)
        final_duration = duration_seconds(final_output)
        oss_key, url = upload(final_output, task_id)
        return jsonify({
            "ok": True,
            "request_id": hashlib.sha1(f"{task_id}:{started}".encode()).hexdigest()[:20],
            "task_id": task_id,
            "model": "MuseTalk",
            "video_url": url,
            "oss_key": oss_key,
            "runtime_ms": int((time.time() - started) * 1000),
            "input_file_size_mb": round(video_size / 1024 / 1024, 3),
            "audio_file_size_mb": round(audio_size / 1024 / 1024, 3),
            "original_duration_seconds": round(source_duration, 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "final_duration_seconds": round(final_duration, 3),
            **info,
        })
    except PermissionError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 401
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc), "task_id": task_id}), 500
    finally:
        if work_dir and os.environ.get("KEEP_WORK_DIR", "0") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)
