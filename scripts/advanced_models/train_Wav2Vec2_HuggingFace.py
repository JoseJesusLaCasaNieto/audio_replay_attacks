"""
Wav2Vec2 Replay Attack Detection using HuggingFace Transformers
Equivalent to the SpeechBrain implementation for binary classification
(genuine vs replay)
"""
import os
import json
import math
import torch
import mlflow
import librosa
import numpy as np

from loguru import logger
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve
)


# Configuration
class ModelConfig:
    """Configuration for the Wav2Vec2 model training"""

    # Model parameters
    model_name: str = "facebook/wav2vec2-base"
    # model_name: str = "facebook/wav2vec2-base-960h"
    num_labels: int = 2  # genuine, replay

    # Training parameters
    num_epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    wav2vec2_learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    logging_steps: int = 500
    save_steps: int = 5000
    # eval_steps: int = 300000
    evaluation_strategy: str = "epoch"
    save_strategy: str = "steps"
    metric_for_best_model: str = "EER"
    save_total_limit: int = 3
    freeze_feature_extractor: bool = True
    freeze_wav2vec2: bool = False

    # Data parameters
    sample_rate: int = 16000
    max_length: int = 16000 * 40  # 40 seconds max

    # Paths
    seed: int = 7
    data_folder: str = "/mnt/media/fair/audio/replay_attacks/datasets/"
    output_folder: str = "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/"
    train_annotation: str = "/mnt/media/fair/audio/replay_attacks/datasets/Mixed_Datasets/ASVSpoof2017_ASVSpoof2019_ReMASC_LRPD_MAILABS_MLAAD_originalandtelephone/Wav2Vec2_annotations/train_annotations_balanced.json"
    valid_annotation: str = "/mnt/media/fair/audio/replay_attacks/datasets/Mixed_Datasets/ASVSpoof2017_ASVSpoof2019_ReMASC_LRPD_MAILABS_MLAAD_originalandtelephone/Wav2Vec2_annotations/dev_annotations_balanced.json"

    # MLFlow configuration
    mlflow_tracking_uri: str = "http://141.94.163.56:5000"
    mlflow_experiment_name: str = "wav2vec2_replay_attacks_hf"
    mlflow_run_name: str = f"Wav2Vec2_HF_Mix{seed}train_{num_epochs}epochs_20251104"
    mlflow_train_artifact_path: str = "configuration_files/train"
    mlflow_model_artifact_path: str = f"model/{seed}"


class ReplayAttackDataset(Dataset):
    """
    Dataset class for replay attack detection
    """

    def __init__(self, annotation_file: str, config: ModelConfig, is_training: bool=True, feature_extractor=None):
        self.config = config
        self.is_training = is_training

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        # Store feature_extractor if provided (not used per-item anymore)
        self.feature_extractor = feature_extractor

        # Create label mapping
        self.label_to_id = {"bonafide": 0, "spoof": 1}  # genuine=0, replay=1

        # Convert dict to list for faster indexing (avoid list(self.data.values()) each time)
        if isinstance(self.data, dict):
            self.items = list(self.data.values())
        else:
            self.items = self.data

        logger.info(f"Loaded {len(self.items)} samples from {annotation_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load audio (return numpy array, not tensor)
        audio_path = item["file_path"]
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")

        # Get label
        label = int(item["spkID"])  # 0 = bonafide, 1 = spoof

        return {
            "audio": audio.astype(np.float32),
            "labels": label
        }


class DataCollatorAudio:
    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, sampling_rate: int = 16000):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, features: List[Dict]):
        # features: list of dicts with keys: 'audio' (1D numpy) and 'labels' (int)
        audios = [f["audio"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        # Let the HF feature_extractor pad/truncate and return batched tensors
        inputs = self.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs["labels"] = labels
        return inputs


def calculate_eer(fpr, tpr):
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def calculate_tDCF(fpr, fnr):
    return float((fpr.mean() + fnr.mean()) / 2)


def compute_metrics(eval_pred):
    """
    eval_pred: (predictions, labels)
        - predictions: logits (N,2) or scores (N,) or probabilities (N,2)
        - labels: (N,)
    Returns: dict with keys: accuracy, f1_score, roc_auc, EER, tDCF
    """
    predictions, labels = eval_pred

    # Ensure numpy arrays
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Get continuous scores and predicted labels
    try:
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            # 2D logits or probabilities -> convert to probabilities via softmax
            exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            scores = probs[:, 1]           # continuous score for the positive class (spoof)
            pred_labels = np.argmax(predictions, axis=1)
        else:
            # 1D predictions: interpret as continuous score
            scores = predictions.ravel()
            # Threshold 0.5 to obtain labels
            pred_labels = (scores >= 0.5).astype(int)
    except Exception:
        # Fallback: if processing fails, return NaNs
        return {
            "accuracy": float('nan'),
            "f1_score": float('nan'),
            "roc_auc": float('nan'),
            "EER": float('nan'),
            "tDCF": float('nan'),
        }

    # Accuracy
    try:
        accuracy = float(accuracy_score(labels, pred_labels))
    except Exception:
        accuracy = float('nan')

    # Classification report -> F1 weighted
    try:
        report = classification_report(labels, pred_labels, output_dict=True, zero_division=0)
        f1_weighted = float(report["weighted avg"]["f1-score"])
    except Exception:
        f1_weighted = float('nan')

    # ROC AUC
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except Exception:
        roc_auc = float('nan')

    # ROC curve and EER
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        eer = calculate_eer(fpr, tpr)
    except Exception:
        eer = float('nan')
        # Ensure fpr/tpr exist for tDCF fallback
        fpr, tpr = None, None

    # tDCF: use your calculate_tDCF function which takes (fpr, fnr=1-tpr)
    try:
        if fpr is not None and tpr is not None:
            fnr = 1.0 - tpr
            tdcf = calculate_tDCF(fpr, fnr)
            tdcf = float(tdcf)
        else:
            tdcf = float('nan')
    except Exception:
        tdcf = float('nan')

    return {
        "accuracy": accuracy,
        "f1_score": f1_weighted,
        "roc_auc": roc_auc,
        "EER": float(eer) if not math.isnan(eer) else float('nan'),
        "tDCF": tdcf,
    }


class CustomTrainer(Trainer):
    """Custom trainer to handle different learning rates for different parts of the model"""

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer with different learning rates for wav2vec2 and classifier"""

        # Separate parameters
        wav2vec2_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if "wav2vec2" in name:
                wav2vec2_params.append(param)
            else:
                classifier_params.append(param)

        # Create optimizer with different learning rates
        optimizer_grouped_parameters = [
            {
                "params": wav2vec2_params,
                "lr": self.args.learning_rate / 10,  # Lower LR for pretrained model
            },
            {
                "params": classifier_params,
                "lr": self.args.learning_rate,
            },
        ]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)

        # Create scheduler
        from transformers.optimization import get_scheduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )


def main():
    """Main training function"""

    # Initialize configuration
    config = ModelConfig()

    # Create output directory
    os.makedirs(config.output_folder, exist_ok=True)

    # Create seed-specific subfolder and set as output_dir
    seed_folder = os.path.join(config.output_folder, str(config.seed))
    os.makedirs(seed_folder, exist_ok=True)

    # Train script directory
    train_script_path = os.path.abspath(__file__)

    # Initialize MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    mlflow.start_run(run_name=config.mlflow_run_name)
    mlflow.log_artifact(
        train_script_path,
        artifact_path=config.mlflow_train_artifact_path
    )

    # Log configuration
    mlflow.log_params({
        "model_name": config.model_name,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "wav2vec2_learning_rate": config.wav2vec2_learning_rate,
        "freeze_feature_extractor": config.freeze_feature_extractor,
    })

    try:
        # Create feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.model_name,
            sampling_rate=config.sample_rate
        )

        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = ReplayAttackDataset(
            config.train_annotation,
            config,
            is_training=True,
            feature_extractor=feature_extractor
        )
        val_dataset = ReplayAttackDataset(
            config.valid_annotation,
            config,
            is_training=False,
            feature_extractor=feature_extractor
        )

        feature_extractor.save_pretrained(seed_folder)
        logger.info(f"Saved feature_extractor to: {seed_folder}")

        # Load model
        logger.info("Loading model...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            ignore_mismatched_sizes=True
        )

        # Freeze feature extractor if specified
        if config.freeze_feature_extractor:
            model.wav2vec2.feature_extractor._freeze_parameters()
            logger.info("Frozen feature extractor")

        # Prepare DataCollator (batch-level processing)
        data_collator = DataCollatorAudio(feature_extractor, sampling_rate=config.sample_rate)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=seed_folder,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            logging_dir=os.path.join(seed_folder, 'logs'),
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            # eval_steps=config.eval_steps,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            load_best_model_at_end=False,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=False,
            report_to="mlflow",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            save_total_limit=config.save_total_limit,
            fp16=True,
            dataloader_pin_memory=True
        )

        # Initialize trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        # Train model
        logger.info("Starting training...")
        try:
            train_result = trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Training from scratch...")
            train_result = trainer.train()

        # Save model
        trainer.save_model()

        # Log training results
        mlflow.log_metrics({f"train_{k}": v for k, v in train_result.metrics.items()})

        # Upload trained model folder to MLFlow artifacts
        logger.info("Uploading model folder to MLFlow artifacts...")
        client = mlflow.tracking.MlflowClient()
        run_id = mlflow.active_run().info.run_id

        client.log_artifacts(
            run_id, seed_folder, artifact_path=config.mlflow_model_artifact_path
        )

        logger.success("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
