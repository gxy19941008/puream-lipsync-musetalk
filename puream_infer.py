import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def quote_yaml(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def ensure_model_link(repo_dir: Path, model_dir: Path) -> None:
    target = repo_dir / "models"
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == model_dir:
            return
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            return
    os.symlink(str(model_dir), str(target), target_is_directory=True)


def write_inference_config(path: Path, video: str, audio: str, output_name: str) -> None:
    path.write_text(
        "\n".join([
            "puream_task:",
            f"  video_path: {quote_yaml(video)}",
            f"  audio_path: {quote_yaml(audio)}",
            f"  result_name: {quote_yaml(output_name)}",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_dir = Path(os.environ.get("MUSETALK_DIR", "/opt/MuseTalk"))
    model_dir = Path(os.environ.get("MODEL_DIR", "/models/MuseTalk"))
    work_dir = Path(os.environ.get("WORK_DIR", "/tmp/puream-lipsync")) / "puream_infer"
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_model_link(repo_dir, model_dir)

    output_path = Path(args.output)
    output_name = output_path.name
    config_path = work_dir / f"{output_path.stem}.yaml"
    write_inference_config(config_path, args.video, args.audio, output_name)

    command = [
        sys.executable,
        str(repo_dir / "scripts" / "inference.py"),
        "--inference_config", str(config_path),
        "--result_dir", str(output_path.parent),
        "--output_vid_name", output_name,
        "--unet_config", str(model_dir / "musetalkV15" / "musetalk.json"),
        "--unet_model_path", str(model_dir / "musetalkV15" / "unet.pth"),
        "--whisper_dir", str(model_dir / "whisper"),
        "--version", "v15",
        "--use_float16",
        "--batch_size", os.environ.get("MUSETALK_BATCH_SIZE", "4"),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_dir}:{existing_pythonpath}" if existing_pythonpath else str(repo_dir)
    completed = subprocess.run(
        command,
        cwd=str(repo_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        return completed.returncode

    produced = output_path.parent / "v15" / output_name
    if not produced.exists() or produced.stat().st_size <= 0:
        print(completed.stdout)
        print(f"Expected output not found: {produced}")
        return 1
    if produced != output_path:
        shutil.copyfile(produced, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
