import json
from loguru import logger

# -------------------------------
# Paths to the input JSON files:
paths = [
    "/mnt/media/fair/audio/replay_attacks/datasets/Mixed_Datasets/ASVSpoof2017_ASVSpoof2019_ReMASC_LRPD_MAILABS_MLAAD_originalandtelephone/Wav2Vec2_annotations/dev_augmented_annotations_balanced.json",
    "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake/Wav2Vec2_annotations/dev_augmented_annotations.json"
]

# Path where you want to save the resulting JSON:
output_path = "/mnt/media/fair/audio/replay_attacks/datasets/Mixed_Datasets/ASVSpoof2017_ASVSpoof2019_ReMASC_LRPD_MAILABS_MLAAD_originalandtelephone_EchoFake/Wav2Vec2_annotations/dev_augmented_annotations_balanced.json"
# ------------------------------


def main():
    # Initialize an empty dictionary to hold merged data from all JSON files
    merged_data = {}

    logger.info("Starting to merge JSON files")
    for path in paths:
        logger.info(f"Reading file: {path}")
        try:
            # Open the current JSON file and load its contents into a Python dict
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"  > Loaded {len(data)} entries from {path}")

            # Merge (or update) all key/value pairs from this JSON into merged_data
            merged_data.update(data)
            logger.info(f"  > Added/updated {len(data)} keys")
        except Exception as e:
            # Log an error if the file could not be read or parsed
            logger.error(f"Error reading {path}: {e}")
    logger.success(f"All files read. Total combined entries: {len(merged_data)}")

    # Save the combined dictionary to output_path as a formatted JSON file
    logger.info(f"Writing combined JSON to: {output_path}")
    try:
        with open(output_path, "w") as f:
            json.dump(merged_data, f, indent=4)
        logger.success(f"Combined JSON file successfully saved to: {output_path}")
    except Exception as e:
        # Log an error if writing to the output file fails
        logger.error(f"Could not write the output file: {e}")


if __name__ == "__main__":
    main()
