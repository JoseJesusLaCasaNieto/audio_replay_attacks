"""
upload_selected_evals_to_run_simple.py

Minimal script to upload several evaluation folders into the ARTIFACTS of an existing MLflow run.
Each evaluation is uploaded under: <basename>/evaluation/... and <basename>/plots/...

Now searches files directly inside each evaluation folder (no longer requires 'evaluation' or 'plots' subfolders).
"""
import os
import json
import traceback
from mlflow.tracking import MlflowClient

# ============ CONFIG =============
TRACKING_URI = "http://141.94.163.56:5000"
TARGET_RUN_ID = "133df99415c94081927bbb2bb0a43c67"

EVAL_FOLDERS = [
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ASVSpoof2017_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ASVSpoof2017telephone_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ASVSpoof2019_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ASVSpoof2019telephone_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_audioreplay_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_LRPD_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_LRPDtelephone_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ReMASC_evaluated",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/evaluate_results/Mix35_trained_Mix35_val_ReMASCtelephone_evaluated"
]
# expected file names (now searched directly inside each eval folder)
METRICS_NAME = "metrics.json"
EVAL_RESULTS_NAME = "evaluation_results.json"
CM_CANDIDATES = ["confusion_matrix.json", "confusion_matrix.txt", "confusion_matrix.csv", "confusion_matrix"]
PLOT_CANDIDATES = ["roc_curve.png", "score_histogram.png"]
# ==================================

client = MlflowClient(tracking_uri=TRACKING_URI)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def log_metrics_from_metrics_json(run_id, metrics_path, prefix):
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        print(f"  Warning: couldn't read metrics file: {metrics_path}")
        traceback.print_exc()
        return
    for k, v in metrics.items():
        val = safe_float(v)
        if val is not None:
            client.log_metric(run_id, f"{prefix}_{k}", val)
            print(f"    logged metric: {prefix}_{k} = {val}")


def log_classification_report(run_id, eval_results_path, prefix):
    try:
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)
    except Exception:
        print(f"  Warning: couldn't read evaluation_results file: {eval_results_path}")
        traceback.print_exc()
        return
    # roc_auc_score
    roc = eval_results.get("roc_auc_score")
    if roc is not None:
        val = safe_float(roc)
        if val is not None:
            client.log_metric(run_id, f"{prefix}_roc_auc_score", val)
            print(f"    logged metric: {prefix}_roc_auc_score = {val}")
    # classification_report flatten
    class_report = eval_results.get("classification_report", {})
    for cls, values in class_report.items():
        if isinstance(values, dict):
            cls_norm = str(cls).replace(" ", "_")
            for mname, mval in values.items():
                val = safe_float(mval)
                if val is not None:
                    metric_key = f"{prefix}_{cls_norm}_{mname.replace(' ', '_').replace('-', '_')}"
                    client.log_metric(run_id, metric_key, val)
                    print(f"    logged metric: {metric_key} = {val}")
        else:
            val = safe_float(values)
            if val is not None:
                client.log_metric(run_id, f"{prefix}_{cls}", val)
                print(f"    logged metric: {prefix}_{cls} = {val}")


def parse_and_log_confusion(run_id, cm_path, prefix):
    if not os.path.isfile(cm_path):
        return False
    parsed = False
    # try JSON first
    try:
        with open(cm_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) >= 2:
            a = safe_float(data[0][0]); b = safe_float(data[0][1])
            c = safe_float(data[1][0]); d = safe_float(data[1][1])
            if None not in (a, b, c, d):
                client.log_metric(run_id, f"{prefix}_cm_r0c0", a)
                client.log_metric(run_id, f"{prefix}_cm_r0c1", b)
                client.log_metric(run_id, f"{prefix}_cm_r1c0", c)
                client.log_metric(run_id, f"{prefix}_cm_r1c1", d)
                print(f"    parsed confusion (json) from {cm_path}: {a},{b},{c},{d}")
                parsed = True
    except Exception:
        pass
    if not parsed:
        # try plain text like "1287 11\n2886 9122"
        try:
            with open(cm_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) >= 2:
                p0 = lines[0].split()
                p1 = lines[1].split()
                if len(p0) >= 2 and len(p1) >= 2:
                    a = safe_float(p0[0]); b = safe_float(p0[1])
                    c = safe_float(p1[0]); d = safe_float(p1[1])
                    if None not in (a, b, c, d):
                        client.log_metric(run_id, f"{prefix}_cm_r0c0", a)
                        client.log_metric(run_id, f"{prefix}_cm_r0c1", b)
                        client.log_metric(run_id, f"{prefix}_cm_r1c0", c)
                        client.log_metric(run_id, f"{prefix}_cm_r1c1", d)
                        print(f"    parsed confusion (text) from {cm_path}: {a},{b},{c},{d}")
                        parsed = True
        except Exception:
            pass
    return parsed


def upload_file_to_artifact(run_id, local_file, artifact_subpath):
    try:
        client.log_artifact(run_id, local_file, artifact_path=artifact_subpath)
        print(f"    uploaded: {local_file} -> {artifact_subpath}")
    except Exception:
        print(f"  ERROR uploading file: {local_file} -> {artifact_subpath}")
        traceback.print_exc()


def upload_folder_files_with_skip(run_id, local_folder, artifact_subpath, skip_set):
    """
    Upload files under local_folder (recursively) into artifacts/<artifact_subpath>/...
    skip_set contains absolute paths to skip (already uploaded).
    """
    if not os.path.isdir(local_folder):
        return
    for root, _, files in os.walk(local_folder):
        for fname in files:
            full = os.path.join(root, fname)
            if full in skip_set:
                continue
            rel = os.path.relpath(root, local_folder)
            if rel == ".":
                artifact_path = artifact_subpath
            else:
                artifact_path = os.path.join(artifact_subpath, rel)
            upload_file_to_artifact(run_id, full, artifact_path)
            skip_set.add(full)


def process_eval_folder(run_id, eval_folder):
    if not os.path.isdir(eval_folder):
        print(f"Skipping (not found): {eval_folder}")
        return

    print("Processing:", eval_folder)
    basename = os.path.basename(eval_folder.rstrip("/\\"))
    prefix = basename.replace(" ", "_").replace("-", "_")
    uploaded = set()  # track absolute paths already uploaded

    # 1) Look for metrics.json and evaluation_results.json directly in eval_folder (root)
    metrics_path = os.path.join(eval_folder, METRICS_NAME)
    if os.path.isfile(metrics_path):
        print(f"  Found metrics: {metrics_path}")
        log_metrics_from_metrics_json(run_id, metrics_path, prefix)
        upload_file_to_artifact(run_id, metrics_path, os.path.join(basename, "evaluation", "json"))
        uploaded.add(os.path.abspath(metrics_path))

    eval_results_path = os.path.join(eval_folder, EVAL_RESULTS_NAME)
    if os.path.isfile(eval_results_path):
        print(f"  Found evaluation_results: {eval_results_path}")
        log_classification_report(run_id, eval_results_path, prefix)
        upload_file_to_artifact(run_id, eval_results_path, os.path.join(basename, "evaluation", "json"))
        uploaded.add(os.path.abspath(eval_results_path))

    # 2) Look for confusion matrix candidates directly in eval_folder
    for c in CM_CANDIDATES:
        cm_path = os.path.join(eval_folder, c)
        if os.path.isfile(cm_path):
            print(f"  Found confusion matrix candidate: {cm_path}")
            parsed = parse_and_log_confusion(run_id, cm_path, prefix)
            upload_file_to_artifact(run_id, cm_path, os.path.join(basename, "evaluation", "json"))
            uploaded.add(os.path.abspath(cm_path))
            break

    # 3) Look for plot files directly in eval_folder (common names)
    for p in PLOT_CANDIDATES:
        p_path = os.path.join(eval_folder, p)
        if os.path.isfile(p_path):
            print(f"  Found plot: {p_path}")
            upload_file_to_artifact(run_id, p_path, os.path.join(basename, "plots"))
            uploaded.add(os.path.abspath(p_path))

    # 4) Also consider other image files in the folder (png/jpg/jpeg) as plots
    for fname in os.listdir(eval_folder):
        full = os.path.join(eval_folder, fname)
        if os.path.isfile(full) and full not in uploaded:
            low = fname.lower()
            if low.endswith((".png", ".jpg", ".jpeg")):
                # if not uploaded above, treat as plot
                upload_file_to_artifact(run_id, full, os.path.join(basename, "plots"))
                uploaded.add(os.path.abspath(full))

    # 5) Upload any remaining files (recursively) into <basename>/evaluation/
    #    skipping already uploaded files.
    upload_folder_files_with_skip(run_id, eval_folder, os.path.join(basename, "evaluation"), uploaded)

    # 6) optional tag to indicate uploaded
    try:
        client.set_tag(run_id, f"evaluation.{prefix}.uploaded", "true")
        client.set_tag(run_id, f"evaluation.{prefix}.source", eval_folder)
    except Exception:
        print("  Warning: couldn't set tags (ignored).")


EXTRA_FILES_TO_UPLOAD = [
    "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2/34/hyperparams.yaml",
    "/mnt/media/fair/audio/replay_attacks/modelos/Wav2Vec2/34/train_antispoofing_Wav2Vec2.py",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/scripts/df_inference_wav2vec.py",
    "/home/pepelacasa/MM-ID-FORENSICS_speechbrain/experiment/evaluate_WAV2VEC2.yaml"
]


def upload_extra_files(run_id, files, artifact_base="configuration_files"):
    """
    Upload a list of local files to the given MLflow run under:
        artifacts/<artifact_base>/train/   <- first two files
        artifacts/<artifact_base>/evaluation/ <- remaining files

    The ordering matters: the function will place the first two files into 'train'
    and any subsequent files into 'evaluation'.
    """
    for idx, fpath in enumerate(files):
        if not os.path.isfile(fpath):
            print(f"  SKIP (not found): {fpath}")
            continue
        bn = os.path.basename(fpath)
        # first two files -> train, others -> evaluation
        if idx < 2:
            dest = os.path.join(artifact_base, "train")
        else:
            dest = os.path.join(artifact_base, "evaluation")
        try:
            upload_file_to_artifact(run_id, fpath, dest)
        except Exception:
            print(f"  ERROR uploading extra file: {fpath} -> {dest}")
            traceback.print_exc()


def main():
    print("MLflow tracking:", TRACKING_URI)
    print("Target run:", TARGET_RUN_ID)
    for folder in EVAL_FOLDERS:
        process_eval_folder(TARGET_RUN_ID, folder)
    print("Uploading extra training files...")
    upload_extra_files(TARGET_RUN_ID, EXTRA_FILES_TO_UPLOAD, artifact_base="configuration_files")
    print("All done. Check MLflow UI artifacts for run:", TARGET_RUN_ID)


if __name__ == "__main__":
    main()
