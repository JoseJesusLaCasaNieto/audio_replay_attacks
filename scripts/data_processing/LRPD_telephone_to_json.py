import os
import json
from tqdm import tqdm
from loguru import logger

# ------------------------------------------------------------------------------
#  1. PATH CONFIGURATION
# ------------------------------------------------------------------------------
TELEPHONE_DATASET_DIRS = [
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/source_trn",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/source_val",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/trn_aparts",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/trn_office",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/val_aparts",
]

ORIGINAL_JSON_PATHS = [
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/Wav2Vec2_annotations/train_annotations.json",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/Wav2Vec2_annotations/dev_annotations.json",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/Wav2Vec2_annotations/eval_annotations.json"
]

TELEPHONE_JSON_OUTPUTS = [
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/Wav2Vec2_annotations/train_annotations.json",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/Wav2Vec2_annotations/dev_annotations.json",
    "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/Wav2Vec2_annotations/eval_annotations.json"
]

# To build the “original” path from the telephone path:
ROOT_TELEPHONE = "/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone"
ROOT_ORIGINAL  = "/mnt/media/fair/audio/replay_attacks/datasets/LRPD"

# ------------------------------------------------------------------------------
#  2. LOAD ORIGINAL JSON FILES AND BUILD A MAP path→(UUID, metadata)
# ------------------------------------------------------------------------------
logger.info("Loading original JSON files...")
# Load each JSON into a dict: key=UUID, value={ "file_path":..., "spkID":..., "length":... }
with open(ORIGINAL_JSON_PATHS[0], "r") as f:
    original_data_train = json.load(f)
with open(ORIGINAL_JSON_PATHS[1], "r") as f:
    original_data_dev = json.load(f)
with open(ORIGINAL_JSON_PATHS[2], "r") as f:
    original_data_eval = json.load(f)
logger.success("Original JSON files loaded successfully")

# Now create a reverse map: original_file_path → (uuid, {spkID, length})
def build_path_index(orig_dict):
    path_index = {}
    logger.info(f"Creating path index: {len(orig_dict)} entries")
    for uuid_key, meta in tqdm(orig_dict.items(), total=len(orig_dict), unit="audio"):
        # meta["file_path"] is the absolute path to the original wav file
        path = meta["file_path"]
        path_index[path] = (uuid_key, meta)
    logger.success("Path index created successfully")
    return path_index

original_index_train = build_path_index(original_data_train)
original_index_dev   = build_path_index(original_data_dev)
original_index_eval  = build_path_index(original_data_eval)

# ------------------------------------------------------------------------------
#  3. PREPARE EMPTY DICTIONARIES FOR OUTPUT TELEPHONE JSON FILES
# ------------------------------------------------------------------------------
telephone_data_train = {}
telephone_data_dev   = {}
telephone_data_eval  = {}

# We'll count how many telephone files are NOT found in any original JSON
missing_count = 0

# ------------------------------------------------------------------------------
#  4. RECURSIVELY ITERATE THROUGH EACH TELEPHONE FOLDER (.wav)
# ------------------------------------------------------------------------------
logger.info("Iterating over LRPD_telephone directories...")

for dataset_dir in tqdm(
    TELEPHONE_DATASET_DIRS,
    total=len(TELEPHONE_DATASET_DIRS),
    unit="LRPD_telephone_path"
):
    for root, _, files in os.walk(dataset_dir):
        for audio_filename in files:
            if not audio_filename.lower().endswith(".wav"):
                continue

            telephone_path = os.path.join(root, audio_filename)

            # 4.1. Check that the '_telephone_' suffix exists
            if "_telephone_" not in audio_filename:
                logger.warning(f"Telephone filename without '_telephone_' suffix: {audio_filename}")
                missing_count += 1
                continue

            # 4.2. Extract the “base” name to reconstruct the original file:
            #       E.g. "singing-01-001_telephone_mulaw.wav" → "singing-01-001.wav"
            base_name = audio_filename.split("_telephone_")[0] + ".wav"

            # 4.3. Build the full path of the ORIGINAL file:
            #      - Get the relative path inside “LRPD_telephone”
            rel_tel = os.path.relpath(telephone_path, ROOT_TELEPHONE)
            #      - Get the relative directory (without the telephone filename)
            dir_rel = os.path.dirname(rel_tel)  # e.g. "source_trn/CN-Celeb/data/id00000"
            #      - Combine that directory with the base name (.wav)
            orig_rel = os.path.join(dir_rel, base_name)  # e.g. ".../singing-01-001.wav"
            #      - Build the absolute path inside LRPD (original dataset)
            orig_path = os.path.join(ROOT_ORIGINAL, orig_rel)

            # 4.4. Determine the split by checking in which original index orig_path exists
            if orig_path in original_index_train:
                orig_index  = original_index_train
                target_dict = telephone_data_train
            elif orig_path in original_index_dev:
                orig_index  = original_index_dev
                target_dict = telephone_data_dev
            elif orig_path in original_index_eval:
                orig_index  = original_index_eval
                target_dict = telephone_data_eval
            else:
                missing_count += 1
                continue

            # 4.5. Build the entry using the same UUID as the original
            original_uuid, original_meta = orig_index[orig_path]
            entry = {
                "file_path": telephone_path,
                "spkID":     original_meta["spkID"],
                "length":    original_meta["length"]
            }
            target_dict[original_uuid] = entry

logger.info(f"Number of telephone audios missing from originals (or misformatted): {missing_count}")

# ------------------------------------------------------------------------------
#  5. FINAL STEP: WRITE EACH TELEPHONE JSON (train, dev, eval) TO DISK
# ------------------------------------------------------------------------------
logger.info("Writing telephone JSON files...")
for idx, out_path in enumerate(TELEPHONE_JSON_OUTPUTS):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if idx == 0:
        data_to_save = telephone_data_train
    elif idx == 1:
        data_to_save = telephone_data_dev
    else:
        data_to_save = telephone_data_eval

    with open(out_path, "w") as fout:
        json.dump(data_to_save, fout, indent=4)
    logger.info(f"Telephone JSON created: {out_path}  (entries: {len(data_to_save)})")

logger.success("The entire process has completed successfully.")
