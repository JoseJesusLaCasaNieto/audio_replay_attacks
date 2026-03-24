import os
import json
import uuid
import torchaudio
from loguru import logger
from tqdm import tqdm


def generate_eval_annotations(root_dir, output_json_path):
    """
    Genera un JSON con:
        - file_path: ruta completa al audio
        - spkID: etiqueta 1 (replay)
        - length: número de muestras de audio
    """
    logger.info(f"Scanning directory: {root_dir}")
    # Recopilar todas las rutas de archivos .wav
    wav_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    logger.info(f"Found {len(wav_files)} .wav files")

    annotations = {}
    # Procesar cada archivo con barra de progreso
    for file_path in tqdm(wav_files, desc="Processing WAV files"):
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
                'spkID': 1,
                'length': length
            }
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    # Guardar JSON
    with open(output_json_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    logger.success(f"Saved annotations to {output_json_path}")


if __name__ == "__main__":
    # Ajusta estas rutas si es necesario
    root_dir = '/mnt/media/fair/audio/replay_attacks/datasets/ReplayDF/wav'
    output_json_path = '/mnt/media/fair/audio/replay_attacks/datasets/ReplayDF/Wav2Vec2_annotations/eval_annotations.json'
    generate_eval_annotations(root_dir, output_json_path)
