import json
from tqdm import tqdm
from loguru import logger

# Input and output files
input_file = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake/Wav2Vec2_annotations/dev_annotations.json"
output_file = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake/Wav2Vec2_annotations/dev_augmented_annotations.json"

logger.info(f"Loading annotations from '{input_file}'...")
with open(input_file, 'r') as f:
    orig = json.load(f)
logger.info(f"Loaded {len(orig)} entries.")

expanded = {}
logger.info("Creating all variants posibilites (0-4) for each entry...")
for utt_id, entry in tqdm(orig.items(), desc="Creating variants", unit="audio"):
    for variant in range(5):
        new_entry = entry.copy()
        new_entry["variant"] = variant

        # Create a new unique ID with a suffix indicating the variant
        new_id = f"{utt_id}_variant{variant}"
        expanded[new_id] = new_entry

logger.info(f"Total augmented entries: {len(expanded)}.")

logger.info(f"Writing augmented annotations to '{output_file}'...")
with open(output_file, 'w') as f:
    json.dump(expanded, f, indent=4)
logger.success("Done.")
