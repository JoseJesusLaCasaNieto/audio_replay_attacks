"""
Evaluate a trained Wav2Vec2ForSequenceClassification model (no CLI args).
Edit the constants below (MODEL_DIR and TEST_ANNOTATION) before running.

Outputs:
    - eval_metrics.json
    - predictions.json
    - predictions.csv
saved into OUTPUT_DIR.

Additionally: uploads OUTPUT_DIR and the evaluation script to the provided
MLflow run (if configured).
"""
import os
import csv
import json
import torch
import mlflow
import librosa
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from loguru import logger
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -----------------------
# CONFIGURATION CLASS (edit these)
# -----------------------
class EvalConfig:
    """
    Configuration container for evaluation script.
    Edit fields here instead of top-level constants.
    """
    # Model & data
    seed = 10
    model_dir = f"/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2_HF/{seed}"
    test_annotation = "/mnt/media/fair/audio/replay_attacks/datasets/EchoFake/Wav2Vec2_annotations/eval_annotations.json"
    output_dir = f"../../evaluate_results/Mix{seed}HF_trained_Mix{seed}HF_val_EchoFake_evaluated"

    # Runtime / batching
    sample_rate = 16000          # model trained at 16 kHz
    max_length = 16000 * 40      # 40 seconds max (same used in training)
    batch_size = 4               # default; change if you want (see notes below)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2

    # MLflow settings (edit to point to your tracking server and run)
    mlflow_tracking_uri = "http://141.94.163.56:5000"  # MLflow tracking server
    mlflow_target_run_id = "ae4334544286427baabf94ae6d4703c6"  # existing run id to upload artifacts
    mlflow_eval_artifact_path = "configuration_files/eval"  # path to this eval script in MLFlow artifacts
    output_dir_basename = os.path.basename(output_dir)
    mlflow_eval_results_path = os.path.join("evaluation_results", output_dir_basename)  # path to evaluation results in MLFlow artifacts


# -----------------------
# Metric helpers
# -----------------------
def calculate_eer(fpr, tpr):
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return float(eer)
    except Exception:
        return float('nan')


def calculate_tDCF(fpr, fnr):
    return float((fpr.mean() + fnr.mean()) / 2)


def compute_metrics_from_preds(predictions, labels):
    preds = np.asarray(predictions)
    labels = np.asarray(labels).ravel()

    try:
        if preds.ndim == 2 and preds.shape[1] > 1:
            exp = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            scores = probs[:, 1]
            pred_labels = np.argmax(preds, axis=1)
        else:
            scores = preds.ravel()
            pred_labels = (scores >= 0.5).astype(int)
    except Exception:
        return {
            "accuracy": float('nan'),
            "f1_score": float('nan'),
            "roc_auc": float('nan'),
            "EER": float('nan'),
            "tDCF": float('nan'),
        }

    try:
        accuracy = float(accuracy_score(labels, pred_labels))
    except Exception:
        accuracy = float('nan')

    try:
        report = classification_report(labels, pred_labels, output_dict=True, zero_division=0)
        f1_weighted = float(report["weighted avg"]["f1-score"])
    except Exception:
        f1_weighted = float('nan')

    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except Exception:
        roc_auc = float('nan')

    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        eer = calculate_eer(fpr, tpr)
    except Exception:
        eer = float('nan')
        fpr, tpr = None, None

    try:
        if fpr is not None and tpr is not None:
            fnr = 1.0 - tpr
            tdcf = calculate_tDCF(fpr, fnr)
        else:
            tdcf = float('nan')
    except Exception:
        tdcf = float('nan')

    return {
        "accuracy": accuracy,
        "f1_score": f1_weighted,
        "roc_auc": roc_auc,
        "EER": eer,
        "tDCF": tdcf,
    }


# -----------------------
# Dataset used for evaluation
# -----------------------
class EvalReplayAttackDataset(Dataset):
    def __init__(self, annotation_file: str, feature_extractor: Wav2Vec2FeatureExtractor,
                 sample_rate: int = 16000, max_length: int = 16000 * 40):
        self.annotation_file = annotation_file
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_length = max_length

        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.meta = [self.data[k] for k in self.keys]
        logger.info(f"Eval dataset loaded: {len(self.meta)} samples from {annotation_file}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]
        audio_path = item["file_path"]
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            audio = np.zeros(self.sample_rate, dtype=np.float32)

        if len(audio) > self.max_length:
            audio = audio[:self.max_length]

        # Keep same label mapping used during training (spkID => 0/1)
        label = int(item.get("spkID"))

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True
        )

        input_values = inputs["input_values"].squeeze(0)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
        else:
            attention_mask = attention_mask.squeeze(0)

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "meta": {
                "file_path": audio_path,
                "orig_label": label,
                "ann_idx_key": self.keys[idx]
            }
        }


def plot_and_save_roc_curve(fpr, tpr, roc_auc_val, roc_curve_file):
    # -----------------------
    # Plot and save ROC curve
    # -----------------------
    try:
        if fpr is not None and tpr is not None:
            logger.info("Plotting and saving ROC curve...")
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (area = {roc_auc_val:.2f})"
            )
            plt.plot(
                [0, 1],
                [0, 1],
                color="navy",
                lw=2,
                linestyle="--"
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_curve_file)
            plt.close()
            logger.info(f"ROC curve saved at: {roc_curve_file}")
        else:
            logger.warning("Cannot plot ROC curve (fpr/tpr not available).")
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")


def plot_and_save_confusion_matrix(cm, confusion_matrix_file):
    # -----------------------
    # Plot and save Confusion Matrix
    # -----------------------
    try:
        if cm is not None:
            logger.info("Plotting and saving confusion matrix...")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(confusion_matrix_file)
            plt.close()
            logger.info(f"Confusion matrix saved at: {confusion_matrix_file}")
        else:
            logger.warning("Cannot plot confusion matrix (cm is None).")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


def upload_results_to_mlflow(
    output_dir: str,
    eval_script_path: str,
    config: EvalConfig
):
    """
    Uploads the evaluation folder and the evaluation script to the specified MLflow run.
    Uses MlflowClient.log_artifacts/log_artifact if mlflow is available.
    """
    if mlflow is None or MlflowClient is None:
        logger.warning("MLFlow or MlflowClient not installed — skipping MLflow upload.")
        return False

    try:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        run_id = config.mlflow_target_run_id

        # Check that run exists (raises if not)
        try:
            client.get_run(run_id)
        except Exception as e:
            logger.error(f"Could not find MLFlow run with ID {run_id}: {e}")
            return False

        # Upload folder with evaluation results
        try:
            logger.info(f"Uploading directory {output_dir} to MLFlow run {run_id} at artifact path '{config.mlflow_eval_results_path}'...")
            client.log_artifacts(run_id, output_dir, artifact_path=config.mlflow_eval_results_path)
            logger.info("Uploaded evaluation directory to MLFlow artifacts.")
        except Exception as e:
            logger.error(f"Failed to upload evaluation directory to MLFlow: {e}")

        # Upload this evaluation script
        try:
            logger.info(f"Uploading eval script {eval_script_path} to MLFlow run {run_id} at artifact path '{config.mlflow_eval_artifact_path}'...")
            client.log_artifact(run_id, eval_script_path, artifact_path=config.mlflow_eval_artifact_path)
            logger.info("Uploaded evaluation script to MLFlow artifacts.")
        except Exception as e:
            logger.error(f"Failed to upload evaluation script to MLFlow: {e}")

        return True
    except Exception as e:
        logger.error(f"Unexpected error while uploading to MLFlow: {e}")
        return False


# -----------------------
# Evaluation routine
# -----------------------
def evaluate(config: EvalConfig):
    logger.info(f"Loading model from {config.model_dir} ...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(config.model_dir)
    logger.success("Model loaded succesfully.")
    try:
        logger.info(f"Loading feature extractor from {config.model_dir}...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_dir)
        logger.success("Loaded feature extractor from model_dir.")
    except Exception:
        try:
            base_model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else None
            if base_model_name:
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name)
                logger.info(f"Loaded feature extractor from base model name {base_model_name}.")
            else:
                feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=config.sample_rate)
                logger.warning("Created default feature extractor.")
        except Exception:
            feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=config.sample_rate)
            logger.warning("Falling back to default feature extractor.")

    device = torch.device(config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu")
    model.to(device)
    model.eval()

    dataset = EvalReplayAttackDataset(config.test_annotation, feature_extractor, sample_rate=config.sample_rate, max_length=config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    all_logits = []
    all_labels = []
    all_scores = []
    all_probs = []
    all_pred_labels = []
    predictions_meta = []

    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            logits_cpu = logits.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            if logits_cpu.ndim == 2 and logits_cpu.shape[1] > 1:
                exp = np.exp(logits_cpu - np.max(logits_cpu, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                scores = probs[:, 1]
                pred_labels = np.argmax(logits_cpu, axis=1)
            else:
                scores = logits_cpu.ravel()
                pred_labels = (scores >= 0.5).astype(int)

            all_logits.append(logits_cpu)
            all_labels.append(labels_cpu)
            all_scores.append(scores)
            all_probs.append(probs)
            all_pred_labels.append(pred_labels)

            batch_meta = batch.get("meta", None)
            for file_path, ann_idx_key, score, pred_lab, label, logit_cpu, probs in zip(batch_meta['file_path'], batch_meta['ann_idx_key'], scores.tolist(), pred_labels.tolist(), labels_cpu.tolist(), logits_cpu.tolist(), probs.tolist()):
                predictions_meta.append({
                    "file_path": file_path,
                    "annotation_variant_key": ann_idx_key,
                    "original_label": int(label),
                    "prediction_label": int(pred_lab),
                    "prediction_score": float(score),
                    "prediction_logits": logit_cpu,
                    "prediction_probabilities": probs
                })

            # Delete unnecessary variables to free up memory
            del logits, logits_cpu, exp, probs, scores, pred_labels, batch_meta
            torch.cuda.empty_cache()

    logits_arr = np.vstack(all_logits) if len(all_logits) else np.zeros((0, ))
    # probs_arr = np.vstack(all_probs) if len(all_probs) else np.zeros((0, ))
    labels_arr = np.concatenate(all_labels) if len(all_labels) else np.zeros((0, ))
    scores_arr = np.concatenate(all_scores) if len(all_scores) else np.zeros((0, ))
    pred_labels_arr = np.concatenate(all_pred_labels) if len(all_pred_labels) else np.zeros((0, ), dtype=int)

    metrics = compute_metrics_from_preds(logits_arr if logits_arr.ndim == 2 else scores_arr, labels_arr)

    # compute ROC curve values to plot (safe-guarded)
    try:
        fpr, tpr, roc_thresh = roc_curve(labels_arr, scores_arr)
    except Exception:
        fpr, tpr, roc_thresh = None, None, None

    try:
        roc_auc_val = float(roc_auc_score(labels_arr, scores_arr))
    except Exception:
        roc_auc_val = float('nan')

    # compute confusion matrix
    try:
        cm = confusion_matrix(labels_arr, pred_labels_arr)
    except Exception:
        cm = None

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(config.output_dir, "eval_metrics.json")
    preds_json_path = os.path.join(config.output_dir, "predictions.json")
    preds_csv_path = os.path.join(config.output_dir, "predictions.csv")
    roc_curve_file = os.path.join(config.output_dir, "roc_curve.png")
    confusion_matrix_file = os.path.join(config.output_dir, "confusion_matrix.png")

    # Enhance metrics with confusion-matrix info and ROC-AUC (already in metrics but we add file paths)
    metrics_enhanced = dict(metrics)
    metrics_enhanced["roc_auc"] = roc_auc_val
    metrics_enhanced["roc_curve_png"] = roc_curve_file
    metrics_enhanced["confusion_matrix_png"] = confusion_matrix_file

    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

    with open(preds_json_path, "w") as fh:
        json.dump(predictions_meta, fh, indent=4)
    logger.info(f"Saved per-sample predictions to {preds_json_path}")

    csv_fields = ["file_path", "annotation_variant_key", "original_label", "prediction_label", "prediction_score", "prediction_logits", "prediction_probabilities"]
    with open(preds_csv_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in predictions_meta:
            writer.writerow({
                "file_path": row.get("file_path"),
                "annotation_variant_key": row.get("annotation_variant_key"),
                "original_label": row.get("original_label"),
                "prediction_label": row.get("prediction_label"),
                "prediction_score": row.get("prediction_score"),
                "prediction_logits": json.dumps(row.get("prediction_logits")),
                "prediction_probabilities": json.dumps(row.get("prediction_probabilities"))
            })
    logger.info(f"Saved predictions CSV to {preds_csv_path}")

    # Plot ROC Curve
    plot_and_save_roc_curve(fpr, tpr, roc_auc_val, roc_curve_file)

    # Plot Confusion Matrix
    plot_and_save_confusion_matrix(cm, confusion_matrix_file)

    # After saving everything locally, attempt to upload to MLFlow run artifacts
    eval_script_path = os.path.abspath(__file__)

    upload_ok = False
    try:
        upload_ok = upload_results_to_mlflow(
                output_dir=config.output_dir,
                eval_script_path=eval_script_path,
                config=config
            )
    except Exception as e:
        logger.error(f"Error during MLFlow upload: {e}")

    if upload_ok:
        logger.success("Results uploaded to MLflow run artifacts.")
    else:
        logger.warning("Results not uploaded to MLflow (see logs).")


# -----------------------
# run
# -----------------------
if __name__ == "__main__":
    logger.info("Starting evaluation. Edit constants at top of file to configure paths.")
    config = EvalConfig()
    evaluate(config=config)
    logger.success("Evaluation finished.")
