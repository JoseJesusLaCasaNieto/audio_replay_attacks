import json
import torchaudio
from loguru import logger
from tqdm import tqdm


def process_audios(json_path):
    """
    Processes audio files listed in a JSON file, logs progress and errors.
    Returns:
        durations: List of each audio duration in seconds.
        max_duration: Maximum duration found.
        count_max: Number of audios exceeding 40 seconds.
    """
    logger.info(f"Loading JSON from {json_path}")
    durations = []

    with open(json_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} audio entries to process")

    # Iterate with progress bar
    for audio_id, info in tqdm(list(data.items()), desc="Processing audios", unit="file"):
        file_path = info.get('file_path')
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            duration_seconds = waveform.shape[1] / sample_rate
            durations.append(duration_seconds)
        except Exception as e:
            logger.error(f"Error processing audio {audio_id} at {file_path}: {e}")

    if durations:
        max_duration = max(durations)
        count_max = sum(1 for dur in durations if dur > 40)
        logger.success(f"Max duration: {max_duration:.2f}s, Count >40s: {count_max}")
    else:
        max_duration = 0
        count_max = 0
        logger.warning("No valid audio durations found.")

    return durations, max_duration, count_max


if __name__ == "__main__":
    logger.info("Starting audio processing...")

    json_path = "/mnt/media/fair/audio/replay_attacks/datasets/MLAAD_telephone/Wav2Vec2_annotations/train_annotations.json"

    durations, max_duration, count_max = process_audios(json_path)

    logger.info("\nSummary of audio processing:")
    logger.info(f"  - Maximum duration found: {max_duration:.2f} seconds")
    logger.info(f"  - Number of audios longer than 40 seconds: {count_max}")
