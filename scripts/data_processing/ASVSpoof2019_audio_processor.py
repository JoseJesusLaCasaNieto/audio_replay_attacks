"""
This script processes audio files in the ASVSpoof2019 dataset directory. It
first tries to open each file with 'soundfile' (which is generally faster).
If that fails, it resorts to 'librosa' and then resaves the file with
'soundfile' for future compatibility.

This approach helps handle problematic files while still taking advantage of
'soundfile' for most audio files, making large datasets more manageable.
"""
import os
import argparse
import logging
import librosa

import soundfile as sf

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


def process_audio_file(file_path):
    """
    Processes an audio file:
    - Tries to open it with soundfile
    - If it fails, opens it with librosa and saves it with soundfile
    """
    try:
        # Try to open with soundfile
        audio, sampling_rate = sf.read(file_path)
        return True, "soundfile"
    except Exception as e:
        try:
            # If it fails, try with librosa and save with soundfile
            logger.debug(f"\nCould not open {file_path} with soundfile: {e}")
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE)
            sf.write(file_path, audio, SAMPLING_RATE)
            logger.info(f"File repaired: {file_path}")
            return True, "librosa"
        except Exception as e2:
            logger.error(f"\nError processing {file_path}: {e2}")
            return False, "error"


def process_dataset_directory(dataset_path):
    """
    Processes all audio files in the dataset directory
    """
    total_files = 0
    soundfile_opened = 0
    librosa_fixed = 0
    failed_files = 0

    # Traverse the dataset directory
    for root, _, files in os.walk(dataset_path):
        audio_files = [f for f in files if f.endswith(('.flac'))]

        if not audio_files:
            continue

        logger.info(f"\nProcessing {len(audio_files)} files in {root}")
        total_files += len(audio_files)

        for audio_file in tqdm(audio_files):
            file_path = os.path.join(root, audio_file)
            success, method = process_audio_file(file_path)

            if success:
                if method == "soundfile":
                    soundfile_opened += 1
                elif method == "librosa":
                    librosa_fixed += 1
            else:
                failed_files += 1

    logger.info("\nProcess completed:")
    logger.info(f"\nTotal files: {total_files}")
    logger.info(f"Correct files (opened with soundfile): {soundfile_opened}")
    logger.info(f"Repaired files (opened with librosa): {librosa_fixed}")
    logger.info(f"Failed files: {failed_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio processor for ASVSpoof2019"
    )
    parser.add_argument("dataset_path", help="Path to ASVSpoof2019 dataset")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        logger.error(f"\nPath {args.dataset_path} does not exist")
        exit(1)

    logger.info(f"\nStarting file processing in {args.dataset_path}")
    process_dataset_directory(args.dataset_path)
