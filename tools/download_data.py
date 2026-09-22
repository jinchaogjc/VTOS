"""Download the VTOS benchmarks (LVIS-Count, PlantSeg-OOD) from Hugging Face into data/.

    python -m tools.download_data
"""
import os
import shutil

from huggingface_hub import snapshot_download

REPO = "tic26/VTOS-Bench"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = {
    "lvis_count": "data/tasklets/lvis_count",
    "plantseg_ood": "data/tasklets/plantseg_ood",
}


def main():
    src = snapshot_download(REPO, repo_type="dataset")
    for name, dst in LAYOUT.items():
        dst = os.path.join(ROOT, dst)
        shutil.copytree(os.path.join(src, name), dst, dirs_exist_ok=True)
        print(f"{name} -> {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
