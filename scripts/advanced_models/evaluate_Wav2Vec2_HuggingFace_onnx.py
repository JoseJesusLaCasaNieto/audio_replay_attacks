"""
Run inference with a Wav2Vec2 ONNX audio classification model.

This script loads a Wav2Vec2-based model exported to ONNX format and performs
inference on an input audio file. It uses the corresponding Hugging Face
feature extractor for preprocessing and outputs the probability score for
the target class (e.g., replay attack detection).

Expected behavior:
    - Load the ONNX model and feature extractor.
    - Process the input audio file.
    - Run inference and print the predicted probability.

All paths must be configured in the user configuration section.
"""

import torch
import librosa
import numpy as np

from loguru import logger
from transformers import Wav2Vec2FeatureExtractor
from optimum.onnxruntime import ORTModelForAudioClassification

# -----------------------
# USER CONFIGURATION (edit these)
# -----------------------
MODEL_DIR = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/1"  # <- put your trained model folder here
ONNX_MODEL_DIR = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/onnx/1_onnx"

logger.info("Loading ONNX saved model...")
onnx_model = ORTModelForAudioClassification.from_pretrained(ONNX_MODEL_DIR)
logger.success("ONNX model loaded succesfully.")

logger.info(f"Loading feature extractor from {MODEL_DIR}...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
logger.success("Loaded feature extractor from model_dir.")

# Audio processing
# audio_path = "/home/pepelacasa/MM-PR-01568-audio_replay_attacks/audio/test_sine_16k_int16.wav"
audio_path = "/mnt/media/fair/audio/replay_attacks/datasets/ASVSpoof2017/ASVspoof2017_V2_train/ASVspoof2017_V2_train/T_1000003.wav"
# audio_path = "/mnt/media/fair/audio/replay_attacks/datasets/ReMASC/core/train/data/1010206.wav"

audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
inputs = feature_extractor(
    audio,
    sampling_rate=sr,
    return_tensors="pt",
    padding=False,
    truncation=False
)

# ONNX inference
with torch.no_grad():
    outputs = onnx_model(**inputs)
    logits = outputs.logits
    logits_cpu = logits.detach().cpu().numpy()

    if logits_cpu.ndim == 2 and logits_cpu.shape[1] > 1:
        # Apply Softmax and print score
        exp = np.exp(logits_cpu - np.max(logits_cpu, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        scores = probs[:, 1]
        print(scores)
