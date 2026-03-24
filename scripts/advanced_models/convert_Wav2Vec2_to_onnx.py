"""
Convert a trained Wav2Vec2 audio classification model to ONNX format.

This script loads a fine-tuned Wav2Vec2 model from a specified directory,
exports it to ONNX format using Optimum’s ORTModelForAudioClassification,
and saves the converted model to the defined output path.

Expected behavior:
    - Load the feature extractor and model from MODEL_DIR.
    - Export the model to ONNX format using the CPU provider.
    - Save the ONNX model to SAVE_DIR.

All paths must be configured in the user configuration section.
"""

from loguru import logger
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor

# -----------------------
# USER CONFIGURATION (edit these)
# -----------------------
MODEL_DIR = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/0"  # <- put your trained model folder here
SAVE_DIR = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/onnx/0_onnx"

logger.info(f"Loading feature extractor from {MODEL_DIR}...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
logger.success("Loaded feature extractor from model_dir.")

logger.info("Converting Wav2Vec2 model to ONNX...")
ort_model = ORTModelForAudioClassification.from_pretrained(
    MODEL_DIR,
    export=True,
    provider="CPUExecutionProvider"
)
ort_model.save_pretrained(SAVE_DIR)
logger.success("Wav2Vec2 model converted succesfully.")
