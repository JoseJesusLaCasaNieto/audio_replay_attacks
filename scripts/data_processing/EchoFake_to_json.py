"""
Dataset Annotation Generator

Generates JSON annotation files (train, dev, eval) for audio classification
datasets. Each annotation contains file paths, binary labels (0=genuine,
1=replay/fake), and audio lengths in samples.

Processes audio files from dataset partitions, extracts metadata, and
calculates audio properties using torchaudio. Supports combining multiple
partitions into single output files.

Usage: Configure paths in CONFIGURATION section and run the script.
"""
import os
import json
import torchaudio
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Dict, List


# ==================== CONFIGURATION ====================
# Base paths
BASE_DATASET_PATH = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake"

# Partition mappings: output_file -> source_partition_folder
PARTITION_MAPPING = {
    "train_annotations.json": "train",
    "dev_annotations.json": "dev",
    "eval_annotations.json": ["open_set_eval", "closed_set_eval"]
}

# Output directory for JSON files
OUTPUT_DIR = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake/Wav2Vec2_annotations"

# Label mapping
LABEL_MAPPING = {
    "bonafide": 0,
    "replay_bonafide": 1,
    "fake": 0,
    "replay_fake": 1
}
# =======================================================


def load_metadata(metadata_path: str) -> List[Dict]:
    """
    Load metadata JSON file and return the list of dictionaries.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        List of metadata dictionaries
    """
    logger.info(f"Loading metadata from: {metadata_path}")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.success(f"Loaded {len(metadata)} entries from metadata")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise


def get_audio_length(file_path: str) -> int:
    """
    Calculate audio length using torchaudio.

    Args:
        file_path: Path to the audio file

    Returns:
        Length of the audio in samples
    """
    try:
        waveform, _ = torchaudio.load(file_path)
        length = waveform.shape[1]
        return length
    except Exception as e:
        logger.warning(f"Error loading audio {file_path}: {e}")
        return 0


def find_label_in_metadata(filename: str, metadata: List[Dict]) -> int:
    """
    Find the label for a given filename in the metadata.

    Args:
        filename: Name of the audio file
        metadata: List of metadata dictionaries

    Returns:
        Label (0 or 1)

    Raises:
        ValueError: If filename is not found in metadata
        or label is unknown.
    """
    for entry in metadata:
        if entry.get("filename") == filename:
            label_str = entry.get("label")
            if label_str in LABEL_MAPPING:
                return LABEL_MAPPING[label_str]
            else:
                error_msg = f"Unknown label '{label_str}' for file {filename}."
                logger.error(error_msg)
                raise ValueError

    error_msg = f"File {filename} not found in metadata"
    logger.error(error_msg)
    raise ValueError(error_msg)


def process_partition(partition_name: str, audio_folder: str, metadata_path: str) -> Dict:
    """
    Process a single partition and generate annotations.

    Args:
        partition_name: Name of the partition (e.g., 'train', 'dev')
        audio_folder: Path to the folder containing audio files
        metadata_path: Path to the metadata JSON file

    Returns:
        Dictionary with annotations for all audio files
    """
    logger.info(f"Processing partition: {partition_name}")

    # Load metadata
    metadata = load_metadata(metadata_path)

    # Get all audio files in the folder
    audio_folder_path = Path(audio_folder)
    audio_extension = ".mp3"
    audio_files = []

    audio_files.extend(list(audio_folder_path.glob(f"*{audio_extension}")))

    logger.info(f"Found {len(audio_files)} audio files in {audio_folder}")

    annotations = {}

    # Process each audio file
    for audio_path in tqdm(audio_files, desc=f"Processing {partition_name}", unit="file"):
        filename = audio_path.name
        file_path_str = str(audio_path.absolute())

        # Get label from metadata
        spk_id = find_label_in_metadata(filename, metadata)

        # Get audio length
        length = get_audio_length(file_path_str)

        # Add to annotations
        annotations[filename] = {
            "file_path": file_path_str,
            "spkID": spk_id,
            "length": length
        }

    logger.success(f"Processed {len(annotations)} files for partition {partition_name}")
    return annotations


def save_annotations(annotations: Dict, output_path: str) -> None:
    """
    Save annotations to a JSON file.

    Args:
        annotations: Dictionary with annotations
        output_path: Path where to save the JSON file
    """
    logger.info(f"Saving annotations to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=4)
        logger.success(f"Annotations saved successfully")
    except Exception as e:
        logger.error(f"Error saving annotations: {e}")
        raise


def main():
    """
    Main function to generate all annotation files.
    """
    logger.info("Starting Dataset Annotations Generation")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Process each partition
    for output_file, source_partition in PARTITION_MAPPING.items():
        logger.info(f"Generating: {output_file}")

        # Process multiple partitions and combine them
        if isinstance(source_partition, list):
            combined_annotations = {}

            for partition in source_partition:
                logger.info(f"Processing sub-partition: {partition}")

                # Build paths
                audio_folder = os.path.join(BASE_DATASET_PATH, partition, "audio")
                metadata_file = os.path.join(BASE_DATASET_PATH, partition, f"{partition}_metadata.json")

                # Check if paths exist
                if not os.path.exists(audio_folder):
                    logger.error(f"Audio folder does not exist: {audio_folder}")
                    continue

                if not os.path.exists(metadata_file):
                    logger.error(f"Metadata file does not exist: {metadata_file}")
                    continue

                # Process partition
                try:
                    annotations = process_partition(partition, audio_folder, metadata_file)

                    # Merge into combined annotations
                    combined_annotations.update(annotations)

                except Exception as e:
                    logger.error(f"Error processing partition {partition}: {e}")
                    continue

            # Save combined annotations
            if combined_annotations:
                output_path = os.path.join(OUTPUT_DIR, output_file)
                save_annotations(combined_annotations, output_path)
            else:
                logger.warning(f"No annotations generated for {output_file}")

        # Process single partition
        else:
            # Build paths
            audio_folder = os.path.join(BASE_DATASET_PATH, source_partition, "audio")
            metadata_file = os.path.join(BASE_DATASET_PATH, source_partition, f"{source_partition}_metadata.json")

            # Check if paths exist
            if not os.path.exists(audio_folder):
                logger.error(f"Audio folder does not exist: {audio_folder}")
                continue

            if not os.path.exists(metadata_file):
                logger.error(f"Metadata file does not exist: {metadata_file}")
                continue

            # Process partition
            try:
                annotations = process_partition(source_partition, audio_folder, metadata_file)

                # Save annotations
                output_path = os.path.join(OUTPUT_DIR, output_file)
                save_annotations(annotations, output_path)

            except Exception as e:
                logger.error(f"Error processing partition {source_partition}: {e}")
                continue

    logger.success("Dataset Annotations Generation Completed")


if __name__ == "__main__":
    main()
