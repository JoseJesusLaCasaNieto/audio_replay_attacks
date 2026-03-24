import os
import json
import uuid
import random
import torchaudio
from loguru import logger
from tqdm import tqdm

def generate_annotations(file_paths, output_json_path):
    """
    Generates a JSON annotation file containing:
        - file_path: full path to the audio
        - spkID: speaker ID (0 for original)
        - length: number of audio samples
    """
    annotations = {}
    for file_path in tqdm(file_paths, desc=f"Processing {os.path.basename(output_json_path)}"):
        try:
            waveform, _ = torchaudio.load(file_path)
            length = waveform.shape[1]
            # Get base filename without extension
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            # Generate unique identifier
            unique_id = uuid.uuid4().hex[:8]
            # Create new filename with unique identifier
            filename = f"{base_name}_{unique_id}{ext}"
            annotations[filename] = {
                'file_path': file_path,
                'spkID': 0, # Original label
                'length': length
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    logger.success(f"Saved annotations to {output_json_path}")


def main(root_dir, train_json_path, dev_json_path, dev_ratio=0.2, seed=42):
    """
    Scans the directory for WAV files, splits them into train and dev sets,
    and generates JSON annotations for each.

    Args:
        root_dir (str): Root directory to scan for .wav files.
        train_json_path (str): Output path for train annotations JSON.
        dev_json_path (str): Output path for dev annotations JSON.
        dev_ratio (float): Proportion of data to assign to dev set.
        seed (int): Random seed for reproducibility.
    """
    logger.info(f"Scanning directory: {root_dir}")
    wav_files = []
    for current_root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(current_root, file))
    logger.info(f"Found {len(wav_files)} .wav files")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * (1 - dev_ratio))
    train_files = wav_files[:split_idx]
    dev_files = wav_files[split_idx:]
    logger.info(f"Assigning {len(train_files)} files to train and {len(dev_files)} files to dev")

    # Generate annotations
    generate_annotations(train_files, train_json_path)
    generate_annotations(dev_files, dev_json_path)


if __name__ == "__main__":
    # Adjust these paths if needed
    root_dir = '/mnt/media/fair/audio/replay_attacks/datasets/MAILABS'
    train_json_path = '/mnt/media/fair/audio/replay_attacks/datasets/MAILABS/Wav2Vec2_annotations/train_annotations.json'
    dev_json_path = '/mnt/media/fair/audio/replay_attacks/datasets/MAILABS/Wav2Vec2_annotations/dev_annotations.json'
    # dev_ratio=0.2 for a 80/20 train/dev split
    main(root_dir, train_json_path, dev_json_path, dev_ratio=0.2)
