import os
import mlflow
from loguru import logger

# Configuration variables
TRACKING_URI = "http://141.94.163.56:5000"
TARGET_RUN_ID = "9ba7b1a4f60b4d678b73b0c364a687eb"
MODEL_PATH = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2/10"
MLFLOW_PATH = "model"


def upload_model_to_mlflow():
    """Upload a local model folder to MLFlow artifacts"""

    # Set MLFlow tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Get the MLFlow client
    client = mlflow.tracking.MlflowClient()

    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

    # Log the folder as artifact
    logger.info(f"Uploading model from {MODEL_PATH} to run {TARGET_RUN_ID}")
    logger.info(f"Target artifact path: {MLFLOW_PATH}")

    client.log_artifact(TARGET_RUN_ID, MODEL_PATH, MLFLOW_PATH)

    logger.success("Upload completed succesfully!")


if __name__ == "__main__":
    upload_model_to_mlflow()
