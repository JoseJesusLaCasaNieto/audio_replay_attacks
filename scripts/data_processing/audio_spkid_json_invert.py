import json
import sys
from loguru import logger
from tqdm import tqdm

# Ruta al archivo JSON de origen (modifica esta ruta según tu necesidad)
FILE_PATH = '/mnt/media/fair/audio/replay_attacks/datasets/ReMASC_telephone/Wav2Vec2_annotations/eval_annotations.json'


def swap_spkIDs(data):
    """
    Intercambia todos los valores de spkID: convierte 1 en 0 y 0 en 1.
    """
    for key, entry in tqdm(data.items(), desc="Procesando entradas", unit="entrada"):
        if 'spkID' in entry:
            original = entry['spkID']
            if original == 1:
                entry['spkID'] = 0
            elif original == 0:
                entry['spkID'] = 1
    return data


def main():
    logger.info(f"Iniciando el procesamiento del archivo: {FILE_PATH}")

    # Carga del JSON original
    try:
        with open(FILE_PATH, 'r') as f:
            content = json.load(f)
        logger.info(f"Archivo cargado correctamente. Entradas totales: {len(content)}")
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo: {FILE_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear JSON: {e}")
        sys.exit(1)

    # Intercambio de etiquetas
    logger.info("Iniciando intercambio de spkID...")
    updated = swap_spkIDs(content)
    logger.success("Intercambio de spkID completado.")

    # Escritura de vuelta al mismo archivo (sobrescribe)
    try:
        with open(FILE_PATH, 'w') as f:
            json.dump(updated, f, indent=4)
        logger.success(f"Archivo actualizado y sobrescrito: {FILE_PATH}")
    except Exception as e:
        logger.error(f"Error al escribir el archivo: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
