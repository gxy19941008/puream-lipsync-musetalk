import os
from huggingface_hub import snapshot_download


def main():
    target = os.environ.get("MODEL_DIR", "/models/MuseTalk")
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=os.environ.get("MUSETALK_HF_REPO", "TMElyralab/MuseTalk"),
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
