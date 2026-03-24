import json
import torchaudio
from loguru import logger
from tqdm import tqdm


def process_and_filter_audios(json_path, threshold_seconds=40):
    """
    Processes audio files listed in a JSON file, logs progress and errors,
    filters out audios exceeding a duration threshold, and saves filtered JSON.
    Returns:
        durations: List of each audio duration in seconds.
        max_duration: Maximum duration found.
        count_exceeding: Number of audios exceeding threshold.
        removed_count: Number of entries removed from JSON.
    """
    logger.info(f"Loading JSON from {json_path}")
    durations = []
    ids_to_remove = []

    with open(json_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Found {len(data)} audio entries to check against threshold of {threshold_seconds}s")

    # Iterate with progress bar
    for audio_id, info in tqdm(list(data.items()), desc="Checking audios", unit="file"):
        file_path = info.get('file_path')
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            duration_seconds = waveform.shape[1] / sample_rate
            durations.append(duration_seconds)

            if duration_seconds > threshold_seconds:
                ids_to_remove.append(audio_id)
        except Exception as e:
            logger.error(f"Error processing audio {audio_id} at {file_path}: {e}")

    # Remove entries exceeding threshold
    for audio_id in ids_to_remove:
        data.pop(audio_id, None)
    removed_count = len(ids_to_remove)
    logger.success(f"Removed {removed_count} entries exceeding {threshold_seconds}s")

    # Save filtered JSON (overwrite)
    save_path = json_path
    with open(save_path, 'w') as f_out:
        json.dump(data, f_out, indent=4)
    logger.info(f"Filtered JSON saved to {save_path}")

    # Summary stats
    max_duration = max(durations) if durations else 0
    count_exceeding = sum(1 for d in durations if d > threshold_seconds)
    logger.success(f"Summary: Max duration {max_duration:.2f}s, Count > threshold {count_exceeding}, Removed {removed_count}")

    return durations, max_duration, count_exceeding, removed_count


if __name__ == "__main__":
    logger.info("Starting processing and filtering of audios...")

    partitions = [
        ("train_annotations.json", "/mnt/media/fair/audio/replay_attacks/datasets/MAILABS_telephone/Wav2Vec2_annotations/train_annotations.json"),
        ("dev_annotations.json", "/mnt/media/fair/audio/replay_attacks/datasets/MAILABS_telephone/Wav2Vec2_annotations/dev_annotations.json"),
    ]

    for name, json_path in partitions:
        logger.info(f"Processing {name}...")
        process_and_filter_audios(json_path, threshold_seconds=40)

    logger.success("\nProcess completed.")
