import os
import json
import torchaudio
from loguru import logger
from tqdm import tqdm


def generate_eval_annotations(root_dir, output_json_path):
    """
    Generates a JSON with:
        - file_path: full path to the audio.
        - spkID: label 1 (replay).
        - length: number of audio samples.
    """
    logger.info(f"Scanning directory: {root_dir}")
    # Collect al .wav file paths
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    logger.info(f"Found {len(wav_files)} .wav files")

    annotations = {}
    for file_path in tqdm(wav_files, desc="Processing WAV files"):
        try:
            waveform, _ = torchaudio.load(file_path)
            length = waveform.shape[1]
            # Get audio filename
            filename = os.path.basename(file_path)
            # Get folder name
            parent_folder = os.path.basename(os.path.dirname(file_path))
            # Determine label
            if parent_folder.lower() == "replay":
                spkID = 1
            else:
                spkID = 0

            annotations[filename] = {
                'file_path': file_path,
                'spkID': spkID,
                'length': length
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    logger.success(f"Saved annotations to {output_json_path}")


if __name__ == "__main__":
    # Adjust paths if necessary
    root_dir = '/mnt/media/fair/audio/replay_attacks/datasets/audio_replay'
    output_json_path = '/mnt/media/fair/audio/replay_attacks/datasets/audio_replay/Wav2Vec2_annotations/eval_annotations.json'
    generate_eval_annotations(root_dir, output_json_path)
