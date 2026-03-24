"""
Copy first 200 audio files described in a JSON metadata file into two folders:
    - original/  (for spkID == 0)
    - replay/    (for spkID == 1)

Configuration (paths) are set near the top of the file.
"""

import json
import shutil
from tqdm import tqdm
from pathlib import Path
from loguru import logger

# ----------------- Configuration -----------------
JSON_PATH = Path("/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019/Wav2Vec2_annotations/eval_annotations.json")   # <-- put the path to your json here
DEST_ORIGINAL = Path("/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019/Spitch/original")  # <-- destination folder for original (spkID == 0)
DEST_REPLAY = Path("/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019/Spitch/replay")      # <-- destination folder for replay (spkID == 1)
N_TO_COPY = 1000
# -------------------------------------------------


def ensure_dirs(*dirs: Path):
    for d in dirs:
        if not d.exists():
            logger.info(f"Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    logger.info(f"Loading JSON file: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def copy_file(src: Path, dst_dir: Path):
    try:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)  # copy2 preserves metadata
        return True, dst
    except Exception as e:
        return False, e


def main():
    logger.info("Script started")
    if not JSON_PATH.exists():
        logger.error(f"JSON file not found: {JSON_PATH}")
        return

    ensure_dirs(DEST_ORIGINAL, DEST_REPLAY)

    data = load_json(JSON_PATH)

    # get first N items in insertion order
    items = list(data.items())[:N_TO_COPY]
    logger.info(f"Found {len(items)} entries to process (requested {N_TO_COPY}).")

    copied = 0
    missing = 0
    failed = 0

    for key, meta in tqdm(items, desc="Copying audios", unit="audio"):
        # meta is expected to be a dict containing at least 'file_path' and 'spkID'
        file_path = meta.get("file_path")
        spkID = meta.get("spkID")

        if file_path is None or spkID is None:
            logger.warning(f"Skipping {key}: missing 'file_path' or 'spkID' in metadata.")
            failed += 1
            continue

        src = Path(file_path)
        if not src.exists():
            logger.warning(f"Source file not found for {key}: {src}")
            missing += 1
            continue

        if int(spkID) == 0:
            dst_dir = DEST_ORIGINAL
        elif int(spkID) == 1:
            dst_dir = DEST_REPLAY
        else:
            logger.warning(f"Unknown spkID ({spkID}) for {key}; skipping.")
            failed += 1
            continue

        ok, result = copy_file(src, dst_dir)
        if ok:
            copied += 1
        else:
            logger.error(f"Failed to copy {src}: {result}")
            failed += 1

    logger.info("Done.")
    logger.info(f"Summary: copied={copied}, missing={missing}, failed={failed}")


if __name__ == "__main__":
    main()
