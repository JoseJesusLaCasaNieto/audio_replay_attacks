"""
LRPD to JSON Converter

This script processes annotation files from the LRPD (Logical Replay Product
Development) dataset and converts them to JSON format compatible with Wav2Vec2
processing requirements.

The script handles three dataset partitions (train, dev, eval) and performs
the following tasks:
    1. Reads annotation text files containing audio file names and their labels
    2. Computes the length (number of samples) of each audio file using
    torchaudio
    3. Creates a structured JSON file for each partition with file paths,
    speaker IDs, and audio lengths
    4. Saves the resulting JSON files to the specified output directory

The JSON structure follows the format:
{
    "filename1.wav": {
        "file_path": "/path/to/audio/file",
        "spkID": 0,  # 0 for bonafide, 1 for spoof
        "length": 12345  # Number of audio samples
    },
    ...
}

Usage:
    python lrpd_to_json.py

Requirements:
    - torchaudio
    - tqdm
    - Access to the LRPD dataset and protocol files
"""
import os
import json
import uuid
import logging
import torchaudio
from tqdm import tqdm
from scripts.LRPD_dataset_splitter import create_directory

# Configuration: adjust these paths as needed
# Base directory where the audio dataset partitions are located
# DATASET_BASE_FOLDER = "/mnt/media/fair/audio/replay_attacks/datasets/LRPD"
DATASET_BASE_FOLDER = "/media/BM/databases/LRPD"

# Directory where the text annotation files are stored
# PROTOCOL_FOLDER = (
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/protocol_V2"
# )
PROTOCOL_FOLDER = (
    "/media/BM/databases/LRPD/protocol_V2"
)

# Output directory for the JSON files
# WAV2VEC2_ANN_FOLDER = (
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/Wav2Vec2_annotations"
# )
WAV2VEC2_ANN_FOLDER = (
    "/media/BM/databases/LRPD/Wav2Vec2_annotations_gea1"
)
create_directory(WAV2VEC2_ANN_FOLDER)

# Dictionary mapping each partition to its corresponding text file name
# PARTITIONS = {
#     "train_annotations": "train_labels.txt",
#     "dev_annotations": "dev_labels.txt",
#     "eval_annotations": "eval_labels.txt"
# }
PARTITIONS = {
    "train_annotations": "train_labels_gea1.txt",
    "dev_annotations": "dev_labels_gea1.txt",
    "eval_annotations": "eval_labels_gea1.txt"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_audio_length(file_path):
    """
    Get the length of the audio file in number of samples using torchaudio.
    """
    try:
        waveform, _ = torchaudio.load(file_path)
        length = waveform.shape[1]
        return length
    except Exception as e:
        logging.error(f"Error loading audio {file_path}: {e}")
        return None


def read_annotation_file(input_annotation_path):
    """
    Read the annotation text file and return its lines.
    """
    logging.info(f"Reading annotation file: {input_annotation_path}")
    try:
        with open(input_annotation_path, 'r') as f:
            lines = f.readlines()
        logging.info("Annotation file read successfully.")
        return lines
    except Exception as e:
        logging.error(f"Error reading file {input_annotation_path}: {e}")
        return None


def process_lines(lines, partition):
    """
    Process each line from the annotation file using tqdm progress bar.
    Constructs and returns the annotations dictionary.
    """
    annotations = {}
    for line in tqdm(
        lines, desc=f"Processing {partition} audios", unit="file"
    ):
        # Each line is expected to contain the filename and the label
        # separated by whitespace
        parts = line.strip().split()
        if len(parts) != 3:
            logging.warning(f"Skipping invalid line: {line.strip()}")
            continue
        _, label, audio_path = parts

        # Get the length of the audio file
        length = get_audio_length(audio_path)
        if length is None:
            logging.warning(f"Skipping file due to error: {audio_path}")
            continue

        # Add the annotation for this audio file
        unique_id = str(uuid.uuid4())
        annotations[unique_id] = {
            "file_path": audio_path,
            "spkID": int(label),
            "length": length
        }
    return annotations


def write_annotations_json(annotations, output_json_path, partition):
    """
    Write the annotations dictionary to a JSON file.
    """
    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(annotations, json_file, indent=4)
        logging.info(
            f"Successfully saved JSON for partition '{partition}' "
            f"at: {output_json_path}"
        )
    except Exception as e:
        logging.error(f"Error writing JSON file {output_json_path}: {e}")


def process_partition(
    partition,
    input_annotation_path,
    output_json_path
):
    """
    Process a dataset partition by reading the annotation file,
    computing the length of each audio, and writing a JSON output.

    Parameters:
    - partition: The partition name (e.g., 'train', 'dev', 'eval')
    - audio_base_path: The base directory where the audios are stored.
    - annotation_file_path: Path to the text file with audio names and labels.
    - output_json_path: Path where the output JSON file will be saved.
    """
    logging.info(f"Processing partition '{partition}'...")

    annotations = {}

    # Read the annotation text file
    lines = read_annotation_file(input_annotation_path)
    if lines is None:
        logging.error(
            f"Error reading annotation file: {input_annotation_path}"
        )
        return

    # Process each line with tqdm progress bar
    annotations = process_lines(lines, partition)

    # Write the annotations dictionary to the JSON file
    write_annotations_json(annotations, output_json_path, partition)

    logging.info(f"Partition {partition} completed.\n")


def main():
    # Process each partition
    logging.info("Starting LRPD dataset to JSON conversion...")
    for partition, txt_filename in PARTITIONS.items():
        input_annotation_path = os.path.join(
            PROTOCOL_FOLDER, txt_filename
        )
        output_json_path = os.path.join(
            WAV2VEC2_ANN_FOLDER, f"{partition}.json"
        )
        process_partition(
            partition=partition,
            input_annotation_path=input_annotation_path,
            output_json_path=output_json_path
        )

    # Log completion message
    logging.info("LRPD dataset to JSON conversion completed.")


if __name__ == "__main__":
    main()
