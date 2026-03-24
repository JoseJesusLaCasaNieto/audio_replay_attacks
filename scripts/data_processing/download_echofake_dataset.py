"""
EchoFake Dataset Audio Downloader
Downloads audio files from the EchoFake dataset on HuggingFace.
Preserves original split structure and saves metadata.
"""
import json
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from datasets import Audio as AudioFeature

# Configuration
OUTPUT_DIR = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake"
DATASET_NAME = "EchoFake/EchoFake"


def setup_output_directory(base_dir: str) -> Path:
    """Create output directory structure."""
    output_path = Path(base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    return output_path


def process_split(dataset, split_name: str, output_dir: Path):
    """
    Process a single dataset split and save audio files with metadata.

    Args:
        dataset: HuggingFace dataset split
        split_name: Name of the split (train, dev, etc.)
        output_dir: Base output directory
    """
    logger.info(f"Processing split: {split_name}")

    # Create split directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Create audio split directory
    split_audio_dir = split_dir / "audio"
    split_audio_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total": len(dataset)
    }

    # Metadata list for all samples in this split
    metadata_list = []

    # Process each sample
    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}", unit="file")):
        try:
            # Extract information
            utt_id = sample.get("utt_id", f"{split_name}_sample_{idx}")
            label = sample.get("label", "unknown")
            audio_data = sample.get("path")

            # Update statistics
            if label not in stats:
                stats[label] = 1
            else:
                stats[label] += 1

            # Get audio array and sampling rate
            audio_bytes = audio_data["bytes"]
            audio_path_original = audio_data.get("path", "")

            # Determine file extension (default to .mp3 if not specified)
            # EchoFake uses MP3 format
            if audio_path_original:
                file_extension = Path(audio_path_original).suffix or ".mp3"
            else:
                file_extension = ".mp3"

            # Save audio file
            output_file = split_audio_dir / f"{utt_id}{file_extension}"
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            # Prepare metadata for this sample
            sample_metadata = {
                "utt_id": utt_id,
                "filename": f"{utt_id}{file_extension}",
                "label": label,
                "source": sample.get("source"),
                "source_text": sample.get("source_text"),
                "source_speaker_id": sample.get("source_speaker_id"),
                "replay_details": sample.get("replay_details"),
                "synthesis_details": sample.get("synthesis_details")
            }

            metadata_list.append(sample_metadata)

        except Exception as e:
            logger.error(f"Error processing sample {idx} (utt_id: {utt_id}): {str(e)}")
            continue

    # Save metadata as JSON file
    metadata_file = split_dir / f"{split_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    logger.success(f"Metadata file saved to: {metadata_file}")

    # Save statistics for this split
    stats_file = split_dir / f"{split_name}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.success(f"Stats file saved to: {stats_file}")


def download_echofake_dataset(output_dir: Path):
    """Download and save audio files from EchoFake dataset."""
    logger.info(f"Loading dataset from HuggingFace: {DATASET_NAME}")

    try:
        # Load the dataset
        dataset_dict = load_dataset(DATASET_NAME)
        logger.success(f"Dataset loaded successfully!")

        # Disable automatic audio decoding to avoid torchcodec/FFmpeg issues
        for split_name in dataset_dict.keys():
            dataset_dict[split_name] = dataset_dict[split_name].cast_column(
                "path",
                AudioFeature(decode=False)
            )

        # Display available splits
        splits = list(dataset_dict.keys())
        logger.info(f"Available splits: {splits}")

        # Process each split
        for split_name in splits:
            split_dataset = dataset_dict[split_name]
            logger.info(f"Split '{split_name}' has {len(split_dataset)} samples")

            # Process the split
            process_split(split_dataset, split_name, output_dir)

    except Exception as e:
        logger.error(f"Fatal error during dataset download:\n{str(e)}")
        logger.exception("Full traceback:")
        raise


def main():
    """Main execution function."""
    logger.info("EchoFake Dataset Audio Downloader")

    # Setup output directory
    output_dir = setup_output_directory(OUTPUT_DIR)

    # Download and process dataset
    download_echofake_dataset(output_dir)

    logger.success("EchoFake dataset download completed successfully!")


if __name__ == "__main__":
    main()
