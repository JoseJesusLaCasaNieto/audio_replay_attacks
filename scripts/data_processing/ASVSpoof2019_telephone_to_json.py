import os
import json
from tqdm import tqdm
from loguru import logger

# Paths
TELEPHONE_DATASET = "/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019_telephone/PA/ASVspoof2019_PA_eval/flac"
ORIGINAL_EVAL_JSON  = "/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019/Wav2Vec2_annotations/eval_annotations.json"
TELEPHONE_EVAL_JSON  = "/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019_telephone/Wav2Vec2_annotations/eval_annotations.json"

# Load the original dummy.json
with open(ORIGINAL_EVAL_JSON, 'r') as f:
    original_data = json.load(f)

telephone_data = {}
count = 0

# Iterate over files in audios_tlf
for audio in tqdm(
    os.listdir(TELEPHONE_DATASET),
    unit="audio"
):
    if not audio.endswith('.wav'):
        logger.info("Audio .wav no encontrado")
        continue

    # Example filename: audio1_tlf_codecXXX.wav
    parts = audio.split('_')
    if len(parts) < 3:
        print(f"Skipping audio with unexpected format: {audio}")
        continue

    base_name = parts[0] + '_' + parts[1] + '_' + parts[2] + '.flac'
    codec = parts[4].replace('.wav', '')

    if base_name not in original_data:
        count += 1
        continue

    # Build new entry
    telephone_data[audio] = {
        "file_path": f"{TELEPHONE_DATASET}/{audio}",
        "spkID": original_data[base_name]["spkID"],
        "length": original_data[base_name]["length"]
    }

logger.info(f"Count = {count}")

# Write to dummy_telephone.json
os.makedirs(os.path.dirname(TELEPHONE_EVAL_JSON), exist_ok=True)
with open(TELEPHONE_EVAL_JSON, 'w') as f:
    json.dump(telephone_data, f, indent=4)

logger.info(f"Created {TELEPHONE_EVAL_JSON} with {len(telephone_data)} entries.")
