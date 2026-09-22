"""Download the VTOS benchmarks (LVIS-Count, PlantSeg-OOD) from Hugging Face into data/.

    python -m tools.download_data
"""
import os
import shutil

# Plain HTTP transfers with a read timeout: a stalled connection raises (and is retried)
# instead of hanging, which the Xet transfer backend can do on unstable networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download  # noqa: E402

REPO = "tic26/VTOS-Bench"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = {
    "lvis_count": "data/tasklets/lvis_count",
    "plantseg_ood": "data/tasklets/plantseg_ood",
}
ATTEMPTS = 5


def main():
    for attempt in range(1, ATTEMPTS + 1):
        try:
            src = snapshot_download(REPO, repo_type="dataset", max_workers=4)
            break
        except Exception as e:  # network errors; finished files are kept in the HF cache
            if attempt == ATTEMPTS:
                raise SystemExit(f"Download failed after {ATTEMPTS} attempts ({type(e).__name__}: {e}).\n"
                                 f"Check the connection to huggingface.co and re-run "
                                 f"`python -m tools.download_data`; finished files are reused.")
            print(f"Download interrupted ({type(e).__name__}); retrying ({attempt}/{ATTEMPTS})")
    for name, dst in LAYOUT.items():
        dst = os.path.join(ROOT, dst)
        shutil.copytree(os.path.join(src, name), dst, dirs_exist_ok=True)
        print(f"{name} -> {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
