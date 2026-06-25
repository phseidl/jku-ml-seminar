"""
scripts/evaluate_checkpoint.py
Score a trained checkpoint on the held-out test ECGs, class by class.

Dataset type and class names come from the result.json that train() saves next
to the checkpoint. For older checkpoints that predate that config-saving, I
fall back to the locked PTB-XL recipe defaults (the original study's strict
recipe).

Usage:
    python -m scripts.evaluate_checkpoint \\
        --checkpoint results/repro_ptbxl_results_matching/s42/checkpoints/best.pt \\
        --out        results/repro_ptbxl_results_matching/eval_s42/eval_result.json \\
        --seed       42

    python -m scripts.evaluate_checkpoint \\
        --checkpoint results/sweep_main/D_georgia_paper_nb2_s42/checkpoints/best.pt \\
        --out        results/sweep_main/eval_georgia_paper_nb2_s42/eval_result.json \\
        --seed       42
"""

import argparse
import json
import os  # noqa: F401  (kept for parity with sibling scripts; not used directly here)
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Make src.* importable no matter where this file is run from. __file__ is
# .../submission/scripts/evaluate_checkpoint.py, so two parents up is the
# project root (.../submission) that holds the src package. Inserting at
# position 0 makes the local package win over any same-named installed one.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fallback data paths, read from config.yaml so the data location lives in one
# place. Used only when a checkpoint's result.json omits the paths; override on
# the command line for a one-off run pointing somewhere else.
_CFG = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
_DEFAULT_PTBXL_DATA_DIR    = str(_CFG.data.data_dir)
_DEFAULT_PTBXL_CACHE_DIR   = str(_CFG.data.cache_dir)
_DEFAULT_GEORGIA_DATA_DIR  = str(_CFG.data.georgia_dir)
_DEFAULT_GEORGIA_CACHE_DIR = str(_CFG.data.georgia_cache_dir)

# Fallback config for checkpoints saved before result.json carried one.
# These are the locked sweep7 PTB-XL hyperparameters. Any checkpoint older than
# config-saving is assumed to have been trained with exactly this
# architecture/preprocessing, so the model can be rebuilt and its labels read
# correctly. The five class names pin the index->class mapping used everywhere
# downstream.
# NOTE: PTB-XL only on purpose -- every Georgia checkpoint postdates config
# saving, so there is no Georgia fallback to keep in sync here.
_SWEEP7_FALLBACK = {
    "dataset_type":  "ptbxl",
    "class_names":   ["NORM", "MI", "STTC", "CD", "HYP"],
    "normalization": "zero_mean_unit_var",
    "embedding_dim": 320,
    "num_blocks":    1,
    "num_heads":     4,
    "dropout":       0.2,
    "pooling":       "mean",
    "fusion_type":   "layer",
    "slstm_backend": "cuda",
    "input_size":    2892,
    "num_classes":   5,
    "data_dir":      _DEFAULT_PTBXL_DATA_DIR,
    "cache_dir":     _DEFAULT_PTBXL_CACHE_DIR,
}


def load_run_config(checkpoint_path: Path) -> dict:
    """
    Recover the training run's config for a given checkpoint.

    Args:
        checkpoint_path: path to a best.pt. The companion result.json is
            expected one directory above the checkpoints/ folder, i.e. at
            <run_dir>/result.json.

    Returns:
        dict: the saved config block from result.json (with data_dir/cache_dir
        backfilled if missing), or a copy of the sweep7 PTB-XL fallback when no
        usable config is found.

    Side effects: prints which source was used (informational, parsed by no one).
    """
    # Layout is <run_dir>/checkpoints/best.pt, so two parents up from the .pt
    # file is the run directory that also holds result.json.
    run_dir     = checkpoint_path.parent.parent   # .../run_name/
    result_file = run_dir / "result.json"

    if result_file.exists():
        with open(result_file) as f:
            saved = json.load(f)
        cfg = saved.get("config")
        if cfg:
            print(f"Config from: {result_file}")
            # Older runs were saved before data_dir/cache_dir were tracked;
            # backfill the PTB-XL defaults so load_test_dataset has somewhere to
            # read from. Georgia runs always save these, so the PTB-XL defaults
            # here are safe -- this only fires on legacy PTB-XL runs.
            if "data_dir" not in cfg:
                cfg["data_dir"]  = _DEFAULT_PTBXL_DATA_DIR
                cfg["cache_dir"] = _DEFAULT_PTBXL_CACHE_DIR
            return cfg
        # result.json exists but carries no config snapshot (very old schema).
        print(f"WARNING: result.json has no 'config' key. Using sweep7 defaults.")
    else:
        print(f"WARNING: No result.json at {result_file}. Using sweep7 defaults.")

    return dict(_SWEEP7_FALLBACK)


def load_model(checkpoint_path: Path, run_cfg: dict, device: torch.device):
    """
    Rebuild XLSTMECGModel from run_cfg and load the trained weights into it.


    Args:
        checkpoint_path: path to the .pt file to load.
        run_cfg: the dict returned by load_run_config(); supplies architecture.
        device: where to build the model and map the checkpoint tensors.

    Returns:
        The XLSTMECGModel in eval mode with weights loaded, on device.

    Raises:
        ValueError: if the checkpoint object is neither a dict nor a bare state
            dict we can load.
    """
    # Imported lazily: omegaconf and the model module (which compiles a CUDA
    # kernel on first import for the sLSTM backend) are heavy. Keeping them out
    # of module scope means a caller that only wants the dataset helpers, say,
    # never pays that cost.
    from omegaconf import OmegaConf
    from src.models.xlstm_ecg import XLSTMECGModel

    # XLSTMECGModel reads its hyperparameters from a nested model section of an
    # OmegaConf node, so I mirror that structure here. Each .get() default is
    # the locked sweep7 value, used only when run_cfg lacks the key (legacy
    # checkpoints). input_size=2892 == 12 leads * 241 STFT freq bins -- it must
    # match the reshape in run_inference and the training loop.
    model_cfg = OmegaConf.create({
        "model": {
            "embedding_dim": run_cfg.get("embedding_dim", 320),
            "num_classes":   run_cfg.get("num_classes",   5),
            "num_blocks":    run_cfg.get("num_blocks",    1),
            "num_heads":     run_cfg.get("num_heads",     4),
            "input_size":    run_cfg.get("input_size",    2892),
            "dropout":       run_cfg.get("dropout",       0.2),
            "slstm_backend": run_cfg.get("slstm_backend", "cuda"),
            "pooling":       run_cfg.get("pooling",       "mean"),
            "fusion_type":   run_cfg.get("fusion_type",   "layer"),
        }
    })

    model = XLSTMECGModel(model_cfg).to(device)

    # map_location=device lets a GPU-saved checkpoint load on CPU (or the other
    # way round) without tripping over a device mismatch. For when the server isn't available.
    ckpt  = torch.load(checkpoint_path, map_location=device)

    # train.py stores the weights under the "model" key (alongside optimizer
    # state, epoch, etc.), so that is the usual path. The other branches make
    # this tolerant of a few conventions from other tooling:
    #   - {"model_state_dict": ...} / {"state_dict": ...}  (Lightning-ish)
    #   - a bare state dict saved directly with torch.save(model.state_dict())
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Weights loaded (key: 'model')")
    elif isinstance(ckpt, dict):

        # Try the alternative wrapper keys; the for/else runs the else only if
        # the loop finishes without break -- i.e. none of the keys were there,
        # so the dict IS itself a state dict.
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                print(f"Weights loaded (key: '{key}')")
                break
        else:
            model.load_state_dict(ckpt)
            print("Weights loaded (direct state dict)")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    model.eval()  # critical: turns dropout off so the scores are deterministic

    # The parameter count is a cheap sanity check that the rebuilt architecture
    # matches the checkpoint. A wrong embedding_dim would already have raised
    # above, but printing the count makes a  fallback-config mismatch
    # visible.
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_test_dataset(run_cfg: dict):
    """
    Build the held-out test split, picking PTB-XL or Georgia from run_cfg.

    Args:
        run_cfg: config dict; dataset_type selects the branch ("ptbxl" |
            "georgia"). The preprocessing knobs read here (label_aggregation,
            nfft, split_strategy, drop_no_target_codes) MUST match training.

    Returns:
        A torch Dataset over the test split.

    Raises:
        ValueError: on an unrecognized dataset_type.
    """
    dataset_type = run_cfg.get("dataset_type", "ptbxl")

    if dataset_type == "ptbxl":
        from src.data.dataset import PTBXLDataset

        # label_aggregation and nfft must match what the model was TRAINED with.
        # Otherwise I'd be grading it against a different label distribution or
        # feature shape than it ever saw -- the numbers would look fine and be
        # wrong. train() embeds both fields in the result.json config snapshot.
        return PTBXLDataset(
            data_dir          = run_cfg.get("data_dir",  _DEFAULT_PTBXL_DATA_DIR),
            split             = "test",
            fs                = 100,   # 100 Hz (filename_lr); the STFT cache and the
                                       # 59-frame sequence length both assume this rate.
            cache_dir         = run_cfg.get("cache_dir", _DEFAULT_PTBXL_CACHE_DIR),

            # label_aggregation "lik_eq_100" keeps only diagnoses the
            # cardiologist asserted with 100% likelihood -- the strict labeling
            # of the original study. Loosen it and a different set of records
            # counts as positive for each class.
            label_aggregation = run_cfg.get("label_aggregation", "lik_eq_100"),

            # nfft default 480 -> 241 freq bins (nfft//2 + 1); the cast guards
            # against nfft arriving as a string from a JSON config snapshot.
            nfft              = int(run_cfg.get("nfft", 480)),
        )

    elif dataset_type == "georgia":
        from src.data.georgia_dataset import GeorgiaECGDataset

        # split_strategy must match training.
        return GeorgiaECGDataset(
            data_dir             = run_cfg.get("data_dir",  _DEFAULT_GEORGIA_DATA_DIR),
            split                = "test",
            cache_dir            = run_cfg.get("cache_dir", _DEFAULT_GEORGIA_CACHE_DIR),
            split_strategy       = run_cfg.get("georgia_split_strategy", "default"),
            drop_no_target_codes = bool(run_cfg.get("georgia_drop_no_target_codes", True)),
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


@torch.no_grad()  # no gradients during eval
def run_inference(model, dataset, device: torch.device,
                  norm_strategy: str, batch_size: int = 512):
    """
    Read every test ECG through the model and collect its labels and scores.

    Mirrors the training pipeline somewhat.

    Args:
        model: an eval-mode XLSTMECGModel (forward() returns sigmoid probs).
        dataset: the test-split Dataset from load_test_dataset().
        device: compute device for the forward pass.
        norm_strategy: normalization name from the run config, passed straight
            to normalize_batch ("zero_mean_unit_var" | "per_channel" | "none").
        batch_size: inference batch size (default 512; large because no grads).

    Returns:
        (labels, scores):
            labels: (N, C) int32   -- ground-truth multi-hot targets
            scores: (N, C) float32 -- per-class probabilities in [0, 1]
        where N = number of test ECGs, C = number of classes.

    Shape walk-through per batch:
        (B, 12, 241, 59) -> permute -> (B, 59, 2892) -> normalize -> model
    """
    from src.training.train import normalize_batch  # reuse the exact training fn

    # shuffle=False keeps the row order fixed across calls.
    # drop_last=False keeps the final partial batch -- I want every test ECG
    # scored, not a round multiple of batch_size.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)
    all_labels, all_scores = [], []
    n = len(loader)

    for i, (x, y) in enumerate(loader):
        # non_blocking pairs with pin_memory above for an async host->device
        # copy; .float() upcasts the cached STFT (it may be float32 already) so
        # the normalization is done in float32, not the kernel's bfloat16.
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        # The dataset emits (B, leads=12, freq=241, frames=59). The model wants
        # a sequence: (B, seq=frames, features=leads*freq). permute(0,3,1,2) ->
        # (B, 59, 12, 241); reshape folds the last two dims into 2892 features
        # per time step. n_frames is the sequence length the xLSTM blocks
        # unroll over -- one step per STFT time window.
        B, n_leads, n_freq, n_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, n_frames, n_leads * n_freq)
        x = normalize_batch(x, norm_strategy)
        # Same NaN guard train.py uses.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        scores = model(x)   # sigmoid already applied inside forward()

        # Move each batch's results to CPU as numpy so GPU memory does not grow
        # with N; I concatenate once at the end rather than scoring in place.
        all_labels.append(y.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

        # Coarse progress print: every 5th batch and always the last one.
        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"  Inference: {i+1}/{n}", flush=True)

    # Stitch the per-batch arrays into the full (N, C) matrices and fix dtypes:
    # labels are integer multi-hot, scores are float probabilities.
    return (np.concatenate(all_labels).astype(np.int32),
            np.concatenate(all_scores).astype(np.float32))


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray,
                        class_names: list[str],
                        threshold: float = 0.5) -> dict:
    """
    Turn (labels, scores) into the full per-class and macro scorecard.

    Args:
        labels: (N, C) int multi-hot ground truth.
        scores: (N, C) float per-class probabilities in [0, 1].
        class_names: length-C names; defines index->class and output ordering.
        threshold: probability cutoff for turning scores into hard yes/no calls
            (default 0.5). AUROC and MAP are threshold-free; the confusion-matrix
            counts -- and everything derived from them -- move with this value.

    Returns:
        dict with macro_auroc, overall_accuracy, precision_macro, recall_macro,
        map, per_class_auroc, per_class_cm, label_cooccurrence, and
        class_distribution. Per-class AUROC is None for any class that is all-0
        or all-1 in labels (AUROC is undefined when only one class is present).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    C    = len(class_names)

    # Hard predictions: >= threshold counts as positive (note >=, so exactly
    # 0.5 calls positive at the default). Multi-label, so each class is its own
    # yes/no -- a single ECG can be positive for several at once and the rows do
    # NOT sum to one.
    preds = (scores >= threshold).astype(np.int32)

    # Macro AUROC = unweighted mean of the per-class AUROC over the (N, C)
    # arrays.
    macro_auroc = float(roc_auc_score(labels, scores, average="macro"))

    # Element-wise match accuracy over the whole (N, C) prediction grid (Hamming accuracy),
    # NOT exact-match-per-ECG accuracy. With many true
    # negatives this number runs high and is the least informative one here.
    overall_acc = float((labels == preds).mean())

    per_class_auroc = {}
    for i, cls in enumerate(class_names):
        # roc_auc_score is undefined when a column holds only one class; check that at least two classes are present and
        # record None rather than let sklearn raise.
        if len(np.unique(labels[:, i])) >= 2:
            per_class_auroc[cls] = round(float(roc_auc_score(labels[:, i], scores[:, i])), 4)
        else:
            per_class_auroc[cls] = None

    per_class_cm = {}
    all_pre, all_sen = [], []   # collected to form the macro precision/recall
    for i, cls in enumerate(class_names):
        yt, yp = labels[:, i], preds[:, i]   # this class's truth / prediction column
        # Standard 2x2 confusion counts for the binary question "is this finding
        # present?".
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        # Each rate guards its denominator against divide-by-zero (a class with
        # no positives gives sen=0 by convention, etc.). The clinical reading:
        # sensitivity is how many true cases we catch, specificity how cleanly
        # we clear the healthy, precision how much to trust a positive call.
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / TPR
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # PPV
        acc = (tp + tn) / (tp + tn + fp + fn)            # denom is N, always > 0

        # F1 = harmonic mean of precision and recall; checked for pre+sen == 0.
        f1  = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0.0
        per_class_cm[cls] = {
            "TP":tp,"TN":tn,"FP":fp,"FN":fn,
            "sensitivity":round(sen,4),"specificity":round(spe,4),
            "precision":round(pre,4),"accuracy":round(acc,4),"f1":round(f1,4),
        }
        all_pre.append(pre); all_sen.append(sen)

    # Label co-occurrence matrices: how often class ii (rows, from a) co-occurs
    # with class jj (cols, from b) in the same ECG (see research notebook for details). true_vs_true shows the
    # multi-label structure of the data -- which diagnoses ride together --
    # while true_vs_predicted shows where the model mistakes one finding for
    # another.
    def _cooccur(a, b, ra, ca):
        mat = np.zeros((C, C), dtype=int)
        for ii in range(C):
            for jj in range(C):
                # Count ECGs where a has class ii AND b has class jj.
                mat[ii, jj] = int(np.sum((a[:, ii] == 1) & (b[:, jj] == 1)))
        # NOTE: O(C^2) python loop, fine for C<=7 here; could be vectorized as
        # a.T @ b if C ever grew large.
        return {"class_names":class_names,"row_axis":ra,"col_axis":ca,"matrix":mat.tolist()}

    # MAP (mean average precision) = macro mean of the per-class average
    # precision, i.e. the area under each precision-recall curve. Needed for the
    # Georgia table.
    try:
        map_score = float(average_precision_score(labels, scores, average="macro"))
    except ValueError:
        # average_precision_score raises if a class has no positive samples;
        # fall back to 0.0 so the report still builds.
        map_score = 0.0

    return {
        "macro_auroc":       round(macro_auroc, 4),
        "overall_accuracy":  round(overall_acc, 4),
        # Macro precision/recall = unweighted mean over classes (every class
        # counts equally, no matter how rare), from the per-class rates above.
        # The macro view keeps a rare-but-serious.
        "precision_macro":   round(float(np.mean(all_pre)), 4),
        "recall_macro":      round(float(np.mean(all_sen)), 4),
        "map":               round(map_score, 4),
        "per_class_auroc":   per_class_auroc,
        "per_class_cm":      per_class_cm,
        "label_cooccurrence": {
            "true_vs_true":      _cooccur(labels, labels, "true", "true"),
            "true_vs_predicted": _cooccur(labels, preds,  "true", "predicted"),
        },

        # Positive/negative support per class -- the denominator behind every
        # rate above, and the explanation for any None per-class AUROC (a class
        # with n_positive == 0 or n_negative == 0 has no AUROC to report).
        "class_distribution": {
            cls: {"n_positive": int(labels[:, i].sum()),
                  "n_negative": int((labels[:, i] == 0).sum())}
            for i, cls in enumerate(class_names)
        },
    }
    # NOTE: F1 is computed per class (per_class_cm[...]["f1"]) but no macro-F1 is
    # surfaced at the top level; add "f1_macro" (mean of the per-class F1s) for
    # symmetry with precision_macro/recall_macro.


def parse_args():
    """Parse the CLI arguments for the standalone evaluation harness.

    --checkpoint and --out are required; the rest tune determinism (--seed),
    the decision cutoff (--threshold), throughput (--batch-size), and device
    placement (--device, "auto" picks CUDA when available else CPU).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)   # path to best.pt
    p.add_argument("--out",        required=True, type=Path)   # eval_result.json path
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--device",     default="auto")             # "auto" | "cuda" | "cpu" | "cuda:1"
    return p.parse_args()


def main():
    # CLI entry point: load -> infer -> score -> print summary -> write JSON.

    args = parse_args()

    # Seed every RNG that could touch the eval.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    # .resolve() to an absolute path so the result.json lookup, which walks the
    # parents of this path, does not depend on the caller's working directory.
    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    run_cfg = load_run_config(ckpt)

    # normalization and class_names drive inference and the metric labeling;
    # fall back to the sweep7 PTB-XL values to stay in step with _SWEEP7_FALLBACK.
    norm_strategy = run_cfg.get("normalization", "zero_mean_unit_var")
    class_names   = run_cfg.get("class_names",
                                ["NORM","MI","STTC","CD","HYP"])
    print(f"Dataset:    {run_cfg.get('dataset_type','ptbxl')}")
    print(f"Norm:       {norm_strategy}")
    print(f"Classes:    {class_names}")
    print(f"Fusion:     {run_cfg.get('fusion_type','layer')}")

    model   = load_model(ckpt, run_cfg, device)
    dataset = load_test_dataset(run_cfg)
    print(f"Test samples: {len(dataset)}")

    # The core, in two steps: collect (labels, scores) over the test split, then
    # reduce them to the scorecard at the requested decision threshold.
    labels, scores = run_inference(model, dataset, device,
                                   norm_strategy, args.batch_size)
    metrics = compute_all_metrics(labels, scores, class_names, args.threshold)

    print(f"\nMacro AUROC:   {metrics['macro_auroc']}")
    print(f"Accuracy:      {metrics['overall_accuracy']}")
    print(f"Precision:     {metrics['precision_macro']}")
    print(f"Recall:        {metrics['recall_macro']}")
    print(f"MAP:           {metrics['map']}")
    for cls in class_names:
        cm = metrics["per_class_cm"][cls]
        print(f"  {cls:6s}: AUROC={metrics['per_class_auroc'][cls]}"
              f"  Sen={cm['sensitivity']}  Spe={cm['specificity']}"
              f"  TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}")

    # Assemble the on-disk record: provenance (which checkpoint, seed,
    # threshold, normalization, class order, N) plus the full run_config
    # snapshot, then into every metric so the JSON top level mirrors the
    # compute_all_metrics keys.
    result = {
        "checkpoint":    str(ckpt),
        "seed":          args.seed,
        "threshold":     args.threshold,
        "norm_strategy": norm_strategy,
        "class_names":   class_names,
        "n_test_samples": int(labels.shape[0]),
        "run_config":    run_cfg,
        **metrics,
    }

    # Create the output directory tree if needed, then write pretty-printed JSON
    # (indent=2) so the result reads cleanly and diffs well in the repo.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    # Prevents this module (as the siblings do) doing a full evaluation -- main() runs only when this is
    # invoked as a script / -m module.
    main()