import os
import requests
from tqdm import tqdm

# 1. Dictionary of language codes and their download URLs
DATASET_URLS = {
    "de_DE": "https://ics.tau-ceti.space/data/Training/stt_tts/de_DE.tgz",
    "en_UK": "https://ics.tau-ceti.space/data/Training/stt_tts/en_UK.tgz",
    "en_US": "https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz",
    "es_ES": "https://ics.tau-ceti.space/data/Training/stt_tts/es_ES.tgz",
    "it_IT": "https://ics.tau-ceti.space/data/Training/stt_tts/it_IT.tgz",
    "uk_UK": "https://ics.tau-ceti.space/data/Training/stt_tts/uk_UK.tgz",
    "ru_RU": "https://ics.tau-ceti.space/data/Training/stt_tts/ru_RU.tgz",
    "fr_FR": "https://ics.tau-ceti.space/data/Training/stt_tts/fr_FR.tgz",
    "pl_PL": "https://ics.tau-ceti.space/data/Training/stt_tts/pl_PL.tgz",
}

# 2. Directory where the files will be saved
OUTPUT_DIR = "/mnt/media/fair/audio/replay_attacks/datasets/MAILABS"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_file(language_folder, url, output_dir):
    local_filename = os.path.join(output_dir, f"{language_folder}.tgz")
    # Stream the request
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        # Download in 1 MB chunks
        with open(local_filename, 'wb') as f, tqdm(
            total=total, unit='iB', unit_scale=True,
            desc=f"{language_folder}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                size = f.write(chunk)
                bar.update(size)


if __name__ == "__main__":
    for language_folder, url in DATASET_URLS.items():
        try:
            print(f"\nDownloading {language_folder} from {url}")
            download_file(language_folder, url, OUTPUT_DIR)
        except Exception as e:
            print(f"Error downloading {language_folder}: {e}")
