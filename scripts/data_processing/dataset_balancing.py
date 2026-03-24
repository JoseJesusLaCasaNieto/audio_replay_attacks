import json
import random
from loguru import logger

# =====================
# Configuration
# =====================
input_paths = {
    'ASVSpoof2017': '/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2017_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
    'ASVSpoof2019_telephone': '/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
    'LRPD_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
    'ReMASC_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/ReMASC_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
    'MAILABS_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/MAILABS_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
    'MLAAD_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/MLAAD_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations.json',
}

output_paths = {
    'ASVSpoof2019_telephone': '/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2019_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations_balanced.json',
    'LRPD_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/LRPD_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations_balanced.json',
    'ReMASC_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/ReMASC_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations_balanced.json',
    'MAILABS_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/MAILABS_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations_balanced.json',
    'MLAAD_telephone':    '/mnt/media/fair/audio/replay_attacks/datasets/MLAAD_telephone/Wav2Vec2_annotations/dev_augmented_random_annotations_balanced.json',
}

# Reproducibility
random.seed(42)


def normalize_items(samples, ds_name):
    """
    Normalize to list of (utt_id, meta) tuples regardless of whether the JSON
    was a dict or a list of dicts.
    """
    if isinstance(samples, dict):
        items = list(samples.items())
        return items
    elif isinstance(samples, list):
        # build items from list
        items = []
        for s in samples:
            utt = s.get('utt_id') or s.get('filename') or None
            if utt is None:
                raise KeyError(f"No utt_id field in sample for {ds_name}")
            items.append((utt, {k: v for k, v in s.items() if k != 'utt_id'}))
        return items
    else:
        raise TypeError(f"Dataset {ds_name} JSON must be dict or list.")


def main():
    # Load ASVSpoof2017 and count
    logger.info("Loading ASVSpoof2017 data from {}", input_paths['ASVSpoof2017'])
    with open(input_paths['ASVSpoof2017'], 'r') as f:
        data_2017 = json.load(f)

    total_ref = len(data_2017)
    if total_ref % 2 != 0:
        logger.warning("Total samples in ASVSpoof2017 is odd ({}), dropping last sample to make even.", total_ref)
        total_ref -= 1
    half = total_ref // 2
    logger.info("ASVSpoof2017 has {} samples, sampling {} per class.", total_ref, half)

    # For each other dataset
    for ds, path in input_paths.items():
        if ds == 'ASVSpoof2017':
            continue

        logger.info("Processing {} from {}", ds, path)
        with open(path, 'r') as f:
            samples = json.load(f)
        items = normalize_items(samples, ds_name=ds)

        # Separate by class
        class0 = [(utt, meta) for utt, meta in items if meta.get('spkID') == 0]
        class1 = [(utt, meta) for utt, meta in items if meta.get('spkID') == 1]
        logger.info("{} has {} class0 and {} class1 samples.", ds, len(class0), len(class1))

        if len(class0) >= half and len(class1) >= half:
            # Sample per class and combine
            sampled = random.sample(class0, half) + random.sample(class1, half)
            logger.info(f"Both classes have >= {half}: sampled {half} from each.")
        elif len(class0) == 0 and len(class1) >= total_ref:
            sampled = random.sample(class1, total_ref)
            logger.info(f"Original class == 0: sampled {total_ref} from replay class")
        elif len(class1) == 0 and len(class0) >= total_ref:
            sampled = random.sample(class0, total_ref)
            logger.info(f"Replay class == 0: sampled {total_ref} from original class")
        else:
            total_available = len(class0) + len(class1)
            logger.error(
                "Dataset {} doesn't meet sampling conditions: need {} total ({} per class or one class == 0 and other >= {}). Available total: {} (class0 {}, class1 {}).",
                ds, total_ref, half, total_ref, total_available, len(class0), len(class1)
            )
            raise ValueError(f"Dataset {ds} cannot provide {total_ref} samples under the simplified rules.")

        # Final shuffle and save
        random.shuffle(sampled)

        # Convert to dict keyed by utt_id
        balanced_dict = {utt: meta for utt, meta in sampled}

        # Write output
        out_file = output_paths.get(ds)
        if not out_file:
            logger.error("No output path defined for {}.", ds)
            raise KeyError(f"No output path for {ds}.")
        logger.info("Writing {} balanced samples to {}", len(balanced_dict), out_file)
        with open(out_file, 'w') as f:
            json.dump(balanced_dict, f, indent=4)

    logger.success("Datasets balanced and saved successfully.")


if __name__ == '__main__':
    main()
