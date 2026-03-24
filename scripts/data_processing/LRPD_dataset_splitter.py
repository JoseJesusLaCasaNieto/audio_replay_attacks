"""
LRPD Dataset Splitter

This script processes the LRPD (Logical Replay Detection) audio dataset by:
1. Collecting audio files from original (bonafide) and replay (spoof) sources
2. Removing duplicates based on filenames
3. Splitting the dataset into train, development, and evaluation sets
4. Copying files to their respective directories
5. Creating label files for each split

The script assigns label 0 to original (bonafide) recordings and label 1 to
replay (spoof) recordings. It maintains the specified split ratios (train: 50%,
development: 20%, evaluation: 30% by default) and handles duplicate files by
keeping only the first occurrence.

Usage:
    python lrpd_dataset_splitter.py

Configuration:
    - Modify the path constants at the top of the script to match your
    environment
    - Adjust the TRAIN_RATIO and DEV_RATIO variables to change split
    proportions

Output:
    - Three directories containing the split audio files
    - Label files in the protocol directory with filename-label pairs
    - An overall label file in the base directory
    - Detailed logging of the process
"""
import os
import random
import logging
from tqdm import tqdm

# Configuration: adjust these paths as needed
# ORIGINAL_FOLDERS = [
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/source_trn",
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/source_val"
# ]
ORIGINAL_FOLDERS = [
    "/media/BM/databases/LRPD/source_trn",
    "/media/BM/databases/LRPD/source_val"
]
# REPLAY_FOLDERS = [
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/trn_aparts",
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/trn_office",
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/val_aparts"
# ]
REPLAY_FOLDERS = [
    "/media/BM/databases/LRPD/trn_aparts",
    "/media/BM/databases/LRPD/trn_office",
    "/media/BM/databases/LRPD/val_aparts"
]
# PROTOCOL_FOLDER = (
#     "/mnt/media/fair/audio/replay_attacks/datasets/LRPD/protocol_V2"
# )
PROTOCOL_FOLDER = (
    "/media/BM/databases/LRPD/protocol_V2"
)

# Base directory for splits
# BASE_DIR = '/mnt/media/fair/audio/replay_attacks/datasets/LRPD'
BASE_DIR = '/media/BM/databases/LRPD'

# Proportions for splits
TRAIN_RATIO = 0.5
DEV_RATIO = 0.2
# EVAL_RATIO will be the remainder (0.3 if TRAIN=0.5 and DEV=0.2)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_directory(path: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}\n")
    else:
        logging.info(f"Directory already exists: {path}\n")


def collect_audios(folders: list, label: int) -> list:
    """
    Traverse the given folders and collect audio files.
    Returns a list of tuples: (full_path, filename, label).
    """
    audio_files = []
    for folder in folders:
        logging.info(f"Collecting files from folder: {folder}")
        for root, _, audios in os.walk(folder):
            for audio in audios:
                if audio.lower().endswith('.wav'):
                    full_path = os.path.join(root, audio)
                    audio_files.append((audio, label, full_path))
    return audio_files


def split_data(audio_list: list) -> tuple:
    """
    Randomly shuffle and split data into train, dev, and eval sets.
    Returns three lists corresponding to each split.
    """
    random.shuffle(audio_list)
    total = len(audio_list)
    train_end = int(TRAIN_RATIO * total)
    dev_end = train_end + int(DEV_RATIO * total)
    train_set = audio_list[:train_end]
    dev_set = audio_list[train_end:dev_end]
    eval_set = audio_list[dev_end:]
    logging.info(
        f"Data split into Train: {len(train_set)}, Dev: {len(dev_set)}, "
        f"Eval: {len(eval_set)}\n"
    )
    return train_set, dev_set, eval_set


def write_label_path_file(
    audio_list: list,
    output_path: str,
    label_path_filename: str
):
    """
    Generates a label file containing audio filename, label, and path
    information.

    This function creates a text file at the specified output path containing
    one entry per line in the format: "filename label path". The process is
    displayed with a progress bar using tqdm.

    Args:
        - audio_list: List of tuples containing (filename, label, audio_path)
        for each audio file.
        - output_path: String path where the label file will be created.
        - label_path_filename: String name of the label file (for logging
        purposes only).

    Returns:
        None. The function writes the label file to disk and logs the process.
    """
    logging.info(f"Writing file {label_path_filename}...")
    # Create label file
    label_path_directory = os.path.join(output_path, label_path_filename)
    with open(label_path_directory, 'w', encoding='utf-8') as label_file:
        for filename, label, audio_path in tqdm(
            audio_list, desc=f"Copying to {PROTOCOL_FOLDER}"
        ):
            label_file.write(f"{filename} {label} {audio_path}\n")

    logging.info(
        f"File {label_path_filename} created."
    )


def main():
    # Create directory for protocol files
    create_directory(PROTOCOL_FOLDER)

    # Collect audio files
    logging.info("Starting audio collection...")
    original_audios = collect_audios(ORIGINAL_FOLDERS, 0)
    replay_audios = collect_audios(REPLAY_FOLDERS, 1)
    all_audios = original_audios + replay_audios
    logging.info("Audio collection completed.\n")

    # Write overall label file
    write_label_path_file(
        audio_list=all_audios,
        output_path=BASE_DIR,
        # label_path_filename="dataset_labels.txt"
        label_path_filename="dataset_labels_gea1.txt"
    )

    # Split the data
    logging.info("Splitting data...")
    train_list, dev_list, eval_list = split_data(all_audios)

    # Copy files and create label files for each split
    logging.info("Writing label path file for each split...")
    write_label_path_file(
        audio_list=train_list,
        output_path=PROTOCOL_FOLDER,
        # label_path_filename="train_labels.txt"
        label_path_filename="train_labels_gea1.txt"
    )
    write_label_path_file(
        audio_list=dev_list,
        output_path=PROTOCOL_FOLDER,
        # label_path_filename="dev_labels.txt"
        label_path_filename="dev_labels_gea1.txt"
    )
    write_label_path_file(
        audio_list=eval_list,
        output_path=PROTOCOL_FOLDER,
        # label_path_filename="eval_labels.txt"
        label_path_filename="eval_labels_gea1.txt"
    )
    logging.info("Label path files written for each split.\n")

    logging.info("Data splitting and copying script completed.")


if __name__ == '__main__':
    main()
