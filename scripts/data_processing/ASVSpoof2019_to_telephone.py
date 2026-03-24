"""
asvspoof2019_to_telephone.py

This script processes the ASVSpoof2019 audio dataset and generates a telephone-band version
of each audio file, using the following steps:

1. Define source and destination dataset directories.
2. Specify three subdirectories (dev, train, eval) to process.
3. Collect all .flac files in each subdirectory.
4. Shuffle the list and split it into two halves: one for G.711 A-law (pcm_alaw) and
    the other for G.711 µ-law (pcm_mulaw).
5. For each file, convert to 8 kHz, mono, 300-3400 Hz bandpass, using the specified codec.
6. Save the converted file in a mirrored folder structure under the destination directory,
    appending "_telephone_alaw" or "_telephone_mulaw" to the filename.
7. Log each major step (start, folder processing, assignments, conversion success/failure)
    using the loguru library.

Usage:
    python asvspoof2019_to_telephone.py

Requirements:
    - Python 3.7+
    - ffmpeg installed and available in PATH
    - loguru installed (pip install loguru)
"""
import os
import subprocess
import random
from pathlib import Path
from loguru import logger

# Parameters: source and destination dataset paths
source_root = Path('/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019')
destination_root = Path('/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019_telephone')

# Specific audio subdirectories to process
SUBDIRS = [
    'PA/ASVspoof2019_PA_dev/flac',
    'PA/ASVspoof2019_PA_train/flac',
    'PA/ASVspoof2019_PA_eval/flac'
]

# Available codecs and labels
CODECS = {
    'pcm_alaw': 'alaw',
    'pcm_mulaw': 'mulaw'
}

# Audio extensions to process
EXTENSIONS = {'.flac'}


def convert_to_telephone(input_path: Path, output_path: Path, codec: str):
    """
    Convert an audio file to telephone-band format (8 kHz, mono, 300–3400 Hz bandpass)
    using the specified codec, and save to output_path.
    """
    try:
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-hide_banner', '-loglevel', 'error',
                '-i', str(input_path),
                '-ar', '8000',
                '-ac', '1',
                '-af', 'highpass=f=300, lowpass=f=3400',
                '-c:a', codec,
                str(output_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        logger.error(f"Conversion failed for {input_path.relative_to(source_root)}")


def main():
    logger.info('==== Starting telephone dataset generation script ====')

    # Verify source directory exists
    if not source_root.exists():
        logger.error(f"Source directory does not exist: {source_root}")
        return

    # Create destination root if missing
    destination_root.mkdir(parents=True, exist_ok=True)

    # Process each specified subdirectory
    for subdir in SUBDIRS:
        src_dir = source_root / subdir
        logger.info(f"Processing directory: {src_dir}")
        if not src_dir.exists():
            logger.warning(f"Directory not found, skipping: {src_dir}")
            continue

        # Gather all audio files
        audio_files = [p for p in src_dir.rglob('*') if p.suffix.lower() in EXTENSIONS]
        logger.info(f"Found {len(audio_files)} files in {src_dir}")
        if not audio_files:
            logger.warning(f"No audio files found in: {src_dir}")
            continue

        # Shuffle and split into two codec groups
        random.shuffle(audio_files)
        half = len(audio_files) // 2
        assignment = {
            'pcm_alaw': audio_files[:half],
            'pcm_mulaw': audio_files[half:]
        }

        # Convert each group
        for codec, files in assignment.items():
            label = CODECS[codec]
            for input_file in files:
                rel = input_file.relative_to(source_root)
                out_dir = destination_root / rel.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                # Construct output filename with suffix
                stem = input_file.stem
                ext = input_file.suffix
                out_file = out_dir / f"{stem}_telephone_{label}.wav"

                convert_to_telephone(input_file, out_file, codec)
        
        logger.info(f"Directory processed succesfully: {src_dir}")

    logger.info('==== Telephone dataset generation completed successfully ====')


if __name__ == '__main__':
    main()
