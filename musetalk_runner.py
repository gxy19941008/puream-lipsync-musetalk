import glob
import json
import os
import subprocess
import sys
from pathlib import Path


class MuseTalkRunner:
    def __init__(self):
        self.repo_dir = Path(os.environ.get("MUSETALK_DIR", "/opt/MuseTalk"))
        self.model_dir = Path(os.environ.get("MODEL_DIR", "/models/MuseTalk"))
        self.result_dir = Path(os.environ.get("WORK_DIR", "/tmp/puream-lipsync")) / "results"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self._models_ready = False

    def run(self, video_path: str, audio_path: str, task_id: str) -> str:
        self._ensure_models()
        output_path = self.result_dir / f"{task_id}_musetalk.mp4"
        cmd = self._command(video_path, audio_path, str(output_path), task_id)
        try:
            subprocess.run(cmd, check=True, cwd=str(self.repo_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "").strip()
            if len(output) > 4000:
                output = output[-4000:]
            raise RuntimeError(f"MuseTalk inference failed: {output}") from exc
        produced = self._pick_output(output_path, task_id)
        if not produced:
            raise RuntimeError("MuseTalk finished but no output video was found")
        if produced != str(output_path):
            subprocess.run(["ffmpeg", "-y", "-i", produced, "-c", "copy", str(output_path)], check=True)
        return str(output_path)

    def _command(self, video_path: str, audio_path: str, output_path: str, task_id: str):
        custom = self.repo_dir / "puream_infer.py"
        if custom.exists():
            return [sys.executable, str(custom), "--video", video_path, "--audio", audio_path, "--output", output_path]

        script = self._find_inference_script()
        args = self._script_args(script)
        cmd = [sys.executable, script]
        if "--video_path" in args:
            cmd += ["--video_path", video_path]
        elif "--video" in args:
            cmd += ["--video", video_path]
        if "--audio_path" in args:
            cmd += ["--audio_path", audio_path]
        elif "--audio" in args:
            cmd += ["--audio", audio_path]
        if "--result_dir" in args:
            cmd += ["--result_dir", str(self.result_dir)]
        if "--output_path" in args:
            cmd += ["--output_path", output_path]
        if "--task_id" in args:
            cmd += ["--task_id", task_id]
        if "--model_dir" in args:
            cmd += ["--model_dir", str(self.model_dir)]
        return cmd

    def _ensure_models(self):
        if self._models_ready or self._model_dir_has_files():
            self._models_ready = True
            return
        fetcher = Path("/app/fetch_models.py")
        if not fetcher.exists():
            raise FileNotFoundError("fetch_models.py not found")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["MODEL_DIR"] = str(self.model_dir)
        subprocess.run([sys.executable, str(fetcher)], check=True, env=env)
        self._models_ready = True

    def _model_dir_has_files(self):
        required = [
            "musetalkV15/musetalk.json",
            "musetalkV15/unet.pth",
            "sd-vae/config.json",
            "sd-vae/diffusion_pytorch_model.bin",
            "whisper/config.json",
            "whisper/pytorch_model.bin",
            "whisper/preprocessor_config.json",
            "dwpose/dw-ll_ucoco_384.pth",
            "face-parse-bisent/79999_iter.pth",
            "face-parse-bisent/resnet18-5c106cde.pth",
        ]
        return self.model_dir.exists() and all((self.model_dir / item).exists() for item in required)

    def _find_inference_script(self) -> str:
        candidates = [
            self.repo_dir / "scripts" / "inference.py",
            self.repo_dir / "inference.py",
            self.repo_dir / "scripts" / "realtime_inference.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError("MuseTalk inference script not found")

    @staticmethod
    def _script_args(script: str):
        try:
            out = subprocess.check_output([sys.executable, script, "--help"], stderr=subprocess.STDOUT, text=True, timeout=30)
        except Exception:
            return set()
        return {part.strip(",") for part in out.split() if part.startswith("--")}

    def _pick_output(self, preferred: Path, task_id: str):
        if preferred.exists() and preferred.stat().st_size > 0:
            return str(preferred)
        patterns = [
            str(self.result_dir / f"*{task_id}*.mp4"),
            str(self.repo_dir / "results" / f"*{task_id}*.mp4"),
            str(self.repo_dir / "results" / "*.mp4"),
        ]
        found = []
        for pattern in patterns:
            found.extend(glob.glob(pattern))
        found = [p for p in found if os.path.getsize(p) > 0]
        return max(found, key=os.path.getmtime) if found else ""
