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


def patch_official_inference(repo_dir: Path) -> None:
    script = repo_dir / "scripts" / "inference.py"
    marker = "# PUREAM continuous-frame patch"
    text = script.read_text(encoding="utf-8")
    if marker in text:
        return

    old_loop = """            # Pad generated images to original video size
            print("Padding generated images to original video size")
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except:
                    continue
                
                # Merge results with version-specific parameters
                if args.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
"""
    new_loop = f"""            # Pad generated images to original video size
            print("Padding generated images to original video size")
            {marker}: keep the image sequence gapless. ffmpeg stops at the first
            # missing %08d.png frame, so a single failed bbox/resize used to truncate
            # the whole output to a few frames.
            for i in tqdm(range(video_num)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                combine_frame = ori_frame
                if bbox != coord_placeholder and i < len(res_frame_list):
                    x1, y1, x2, y2 = bbox
                    if args.version == "v15":
                        y2 = y2 + args.extra_margin
                        y2 = min(y2, ori_frame.shape[0])
                    try:
                        res_frame = cv2.resize(res_frame_list[i].astype(np.uint8), (x2-x1, y2-y1))
                        # Merge results with version-specific parameters
                        if args.version == "v15":
                            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                        else:
                            combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                    except Exception as frame_error:
                        print(f"PUREAM frame fallback {{i}}: {{frame_error}}")
                cv2.imwrite(f"{{result_img_save_path}}/{{str(i).zfill(8)}}.png", combine_frame)
"""
    if old_loop not in text:
        raise RuntimeError("Unable to patch MuseTalk inference frame loop")
    text = text.replace(old_loop, new_loop)

    old_mux = """            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
"""
    new_mux = """            cmd_combine_audio = f"ffmpeg -y -v warning -i {temp_vid_path} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest {output_vid_name}"
"""
    if old_mux not in text:
        raise RuntimeError("Unable to patch MuseTalk audio mux command")
    text = text.replace(old_mux, new_mux)
    script.write_text(text, encoding="utf-8")


def media_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return float(completed.stdout.strip())


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
    patch_official_inference(repo_dir)

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
    expected_duration = min(media_duration(Path(args.video)), media_duration(Path(args.audio)))
    produced_duration = media_duration(output_path)
    if produced_duration + 0.5 < expected_duration * 0.9:
        print(completed.stdout)
        print(
            "Output duration too short: "
            f"expected_about={expected_duration:.3f}s produced={produced_duration:.3f}s"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
