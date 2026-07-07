import os
from huggingface_hub import snapshot_download


def download(repo_id, local_dir, allow_patterns):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )


def main():
    target = os.environ.get("MODEL_DIR", "/models/MuseTalk")
    os.makedirs(target, exist_ok=True)
    download(
        os.environ.get("MUSETALK_HF_REPO", "TMElyralab/MuseTalk"),
        target,
        [
            "musetalk/musetalk.json",
            "musetalk/pytorch_model.bin",
            "musetalkV15/musetalk.json",
            "musetalkV15/unet.pth",
        ],
    )
    download(
        "stabilityai/sd-vae-ft-mse",
        os.path.join(target, "sd-vae"),
        ["config.json", "diffusion_pytorch_model.bin"],
    )
    download(
        "openai/whisper-tiny",
        os.path.join(target, "whisper"),
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )
    download(
        "yzd-v/DWPose",
        os.path.join(target, "dwpose"),
        ["dw-ll_ucoco_384.pth"],
    )
    download(
        "ByteDance/LatentSync",
        os.path.join(target, "syncnet"),
        ["latentsync_syncnet.pt"],
    )
    download(
        "ManyOtherFunctions/face-parse-bisent",
        os.path.join(target, "face-parse-bisent"),
        ["79999_iter.pth", "resnet18-5c106cde.pth"],
    )


if __name__ == "__main__":
    main()
