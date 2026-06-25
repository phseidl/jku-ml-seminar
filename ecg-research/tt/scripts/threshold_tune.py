"""
scripts/threshold_tune.py
=========================
Per-class threshold calibration for the binary-decision metrics
(precision, recall, F1, accuracy); post-training calibration of a frozen checkpoint's outputs.

Usage
-----
.venv/bin/python -m scripts.threshold_tune \\
    --checkpoint results/repro_ptbxl_results_matching/s11/checkpoints/best.pt \\
    --out        results/repro_ptbxl_results_matching/threshold_results_matching_s11.json

Optional flags:
- '--batch-size 512' (default; reduce if you hit out-of-memory)
- '--grid-step 0.01' (default; the F1-vs-threshold curve is smooth, so a
  0.01 grid is plenty fine)

Output JSON schema
------------------
{
  "checkpoint":       "<path>",
  "dataset_type":     "ptbxl" | "georgia",
  "class_names":      [...],
  "n_train":          17111,
  "n_test":           1711,
  "tuned_thresholds": {"NORM": 0.43, "MI": 0.34, ...},
  "baseline_at_0_5": {
    "macro_auroc":  0.9088,
    "map":          0.7365,
    "precision_macro": 0.6932,
    "recall_macro":    0.6541,
    "f1_macro":        0.6680,
    "per_class": {cls: {threshold: 0.5, TP, TN, FP, FN, sensitivity, specificity, precision, f1}}
  },
  "tuned": {  ## same shape as baseline_at_0_5, with tuned thresholds applied
    "macro_auroc":     0.9088,  # unchanged by definition
    "precision_macro": 0.6358,
    "recall_macro":    0.6751,
    "f1_macro":        0.6736,  # the "+DeltaF1" lift
    "per_class": {...}
  }
}

'tuned.f1_macro' minus 'baseline_at_0_5.f1_macro' is the gain in macro F1
from per-class threshold calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_checkpoint import (
    load_run_config,
    load_model,
    run_inference,
)


def _load_split(run_cfg: dict, split: str):
    """Load one split (train or test) for either PTB-XL or Georgia.

    Args:
        run_cfg: The config snapshot read from the checkpoint's companion
            result.json (via 'load_run_config'). Supplies 'dataset_type', the
            data/cache directories, and -- the part that actually matters here
            -- the label-definition knobs that must be reproduced exactly at
            eval time.
        split: '"train"' (used to calibrate the thresholds) or '"test"' (used
            to report the calibrated and baseline metrics). The string passes
            straight through to the dataset class, which maps it to the split
            used in the original study (PTB-XL: train=folds 1-8, val=9, test=10;
            Georgia: test=g1, val=g2, train=g3..g11).

    Returns:
        A 'torch.utils.data.Dataset' that yields '(x, y)' pairs where 'x' is the
        per-lead STFT magnitude tensor '(12, 241, 59)' and 'y' is the multi-hot
        label vector '(C,)'.

    Raises:
        ValueError: if 'run_cfg["dataset_type"]' is neither "ptbxl" nor
            "georgia".
    """

    # Default to "ptbxl" so pre-result.json checkpoints (which lack a
    # dataset_type key) stay on the original PTB-XL path rather than erroring.
    dataset_type = run_cfg.get("dataset_type", "ptbxl")
    if dataset_type == "ptbxl":

        from src.data.dataset import PTBXLDataset
        return PTBXLDataset(
            data_dir          = run_cfg["data_dir"],
            split             = split,
            fs                = 100,                                            # 100 Hz; PTB-XL filename_lr column
            cache_dir         = run_cfg["cache_dir"],
            label_aggregation = run_cfg.get("label_aggregation", "lik_eq_100"),  # MUST match training (see docstring)
            nfft              = int(run_cfg.get("nfft", 480)),                   # 480 -> 241 freq bins (default in the original study)
        )
    if dataset_type == "georgia":
        # Same lazy-import rationale as the PTB-XL branch above.
        from src.data.georgia_dataset import GeorgiaECGDataset
        return GeorgiaECGDataset(
            data_dir             = run_cfg["data_dir"],
            split                = split,
            cache_dir            = run_cfg["cache_dir"],
            split_strategy       = run_cfg.get("georgia_split_strategy", "default"),
            # bool(...) coerces a JSON snapshot's 0/1/"true" into a real Python
            # bool; the dataset class branches on plain truthiness.
            drop_no_target_codes = bool(run_cfg.get("georgia_drop_no_target_codes", True)),
        )

    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


def _f1(tp: int, fp: int, fn: int) -> float:
    """Binary F1 straight from confusion-matrix counts, zero-safe.

    F1 = 2 * precision * recall / (precision + recall), written directly in
    TP/FP/FN so callers never have to assemble precision and recall themselves.

    Args:
        tp: true positives.
        fp: false positives.
        fn: false negatives.

    Returns:
        The F1 score in '[0.0, 1.0]'. Returns '0.0' for the degenerate cases
        that would otherwise divide by zero -- a class with nothing predicted
        positive (precision undefined) or nothing actually positive (recall
        undefined) scores 0.0 rather than raising. This matches sklearn's
        'zero_division=0' convention.
    """
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (pre + rec) == 0 else 2 * pre * rec / (pre + rec)


def find_threshold_per_class(labels: np.ndarray, scores: np.ndarray,
                              grid: np.ndarray) -> list[float]:
    """For each class, return the threshold that maximizes F1 on the train set.

    Args:
        labels: '(N, C)' 0/1 ground-truth matrix (normally the TRAIN split,
            since calibrating on the test set would be circular).
        scores: '(N, C)' sigmoid probabilities, same orientation as 'labels'.
        grid: 1-D array of candidate thresholds to scan (e.g. 0.01..0.99 in
            steps of '--grid-step'). Shared across all classes.

    Returns:
        A list of 'C' floats -- the F1-optimal threshold for each class column,
        in the same column order as 'labels'/'scores'.
    """
    C = labels.shape[1]                      # number of label columns / classes
    best = []
    for c in range(C):
        # One class column at a time: y, p are both shape (N,).
        y, p = labels[:, c], scores[:, c]
        scan = []
        for t in grid:
            # Call this class positive where its probability is at least t.
            # '>=' (not '>') matches the test-time rule in evaluate_at_thresholds
            # and in evaluate_checkpoint -- all three must agree, or a tuned
            # threshold would land one grid-bin off when re-applied.
            yp = (p >= t).astype(int)

            # Confusion counts via boolean masks: AND the two (N,) arrays and
            # sum. np.sum over a bool array counts the Trues; int(...) strips the
            # numpy scalar type so the JSON dump stays clean.
            tp = int(np.sum((y == 1) & (yp == 1)))
            fp = int(np.sum((y == 0) & (yp == 1)))
            fn = int(np.sum((y == 1) & (yp == 0)))

            # Keep (F1, threshold) so I can sort by F1 first, threshold second.
            scan.append((_f1(tp, fp, fn), float(t)))

        # Sort by descending F1 (-x[0]); break ties by ASCENDING threshold
        # (x[1]). The lower-threshold tie-break deliberately leans toward
        # calling positive more often -- on these imbalanced labels the cost of
        # missing a rare positive outweighs one extra false alarm -- and it
        # keeps the choice deterministic across runs.
        scan.sort(key=lambda x: (-x[0], x[1]))
        best.append(scan[0][1])              # take the threshold of the top row

    return best


def evaluate_at_thresholds(labels: np.ndarray, scores: np.ndarray,
                            thresholds: list[float],
                            class_names: list[str]) -> dict:
    """Score TP/FN/FP/TN and per-class + macro precision/recall/F1 at the given thresholds.
    Args:
        labels: '(N, C)' 0/1 ground truth (the TEST split at report time).
        scores: '(N, C)' sigmoid probabilities aligned with 'labels'.
        thresholds: length-'C' list of cutoffs, one per class column. Order must
            match the 'labels'/'scores' columns.
        class_names: length-'C' readable names, used as the keys of the returned
            'per_class' dict (e.g. "NORM", "MI", ...).

    Returns:
        A dict with macro-aggregated AUROC, mAP, precision, recall and F1, plus
        a 'per_class' sub-dict keyed by class name. 'macro_auroc' and 'map' are
        threshold-free, so they come out identical between the baseline and
        tuned calls.
    """
    C = labels.shape[1]
    per_class = {}
    # Accumulators for the macro means -- one entry per class column.
    macro_pre, macro_rec, macro_f1 = [], [], []
    for c in range(C):
        # One class column: y, p are shape (N,); threshold is this class's own.
        y, p = labels[:, c], scores[:, c]

        yp = (p >= thresholds[c]).astype(int)

        tp = int(np.sum((y == 1) & (yp == 1)))
        tn = int(np.sum((y == 0) & (yp == 0)))
        fp = int(np.sum((y == 0) & (yp == 1)))
        fn = int(np.sum((y == 1) & (yp == 0)))

        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1  = _f1(tp, fp, fn)
        per_class[class_names[c]] = {

            "threshold": round(thresholds[c], 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": round(sen, 4), "specificity": round(spe, 4),
            "precision":   round(pre, 4), "f1":          round(f1, 4),
        }
        # macro_rec collects SENSITIVITY (recall == sensitivity for the positive
        # class), NOT specificity.
        macro_pre.append(pre); macro_rec.append(sen); macro_f1.append(f1)

    auroc = float(roc_auc_score(labels, scores, average="macro"))
    map_  = float(average_precision_score(labels, scores, average="macro"))
    return {
        "macro_auroc":     round(auroc, 4),   # identical across baseline vs tuned
        "map":             round(map_, 4),    # ditto -- threshold-independent
        "precision_macro": round(float(np.mean(macro_pre)), 4),
        "recall_macro":    round(float(np.mean(macro_rec)), 4),
        "f1_macro":        round(float(np.mean(macro_f1)),  4),
        "per_class":       per_class,
    }


def main() -> None:
    """CLI entry point: tune per-class thresholds on train, report on test.

    End-to-end flow (each step is visible in the prints it emits):
      1. Parse args and pick the device (CUDA if available, else CPU).
      2. Load the checkpoint's config snapshot and rebuild the model.
      3. Run eval-pipeline inference on the TRAIN split -> calibration scores.
      4. Sweep a threshold grid per class, keeping the F1-optimal threshold.
      5. Run inference on the TEST split, scoring it both at the flat 0.5
         baseline and at the tuned thresholds.
      6. Dump everything (thresholds + both metric blocks) to '--out' as JSON.

    Returns 'None'; the deliverables are the JSON file written to '--out' and
    the summary printed to stdout. Side effects: creates the parent directory of
    '--out' if needed and overwrites any existing file there.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)   # path to best.pt
    p.add_argument("--out",        required=True)    # path to write the result JSON
    p.add_argument("--batch-size", type=int, default=512)   # inference batch; lower if OOM
    p.add_argument("--grid-step",  type=float, default=0.01,
                   help="Threshold grid resolution (default 0.01).")
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt     = Path(args.checkpoint)
    run_cfg  = load_run_config(ckpt)

    norm     = run_cfg.get("normalization", "zero_mean_unit_var")

    classes  = run_cfg["class_names"]

    print(f"Dataset    : {run_cfg.get('dataset_type', 'ptbxl')}")
    print(f"Classes    : {classes}")

    model = load_model(ckpt, run_cfg, device)

    # --- Step 3+4: TRAIN scores -> calibrate thresholds -------------------
    # Calibrate on TRAIN, not TEST: choosing thresholds on the same data I then
    # report on would leak the test set into the decision rule and inflate F1.
    print("\nLoading train split for threshold calibration...")
    train_ds = _load_split(run_cfg, "train")
    print(f"  train N = {len(train_ds)}")
    print("Running inference on train split...")
    # L_tr: (N_train, C) int labels; S_tr: (N_train, C) float sigmoid scores.
    L_tr, S_tr = run_inference(model, train_ds, device, norm, args.batch_size)

    # Grid = grid_step, 2*grid_step, ..., up to but excluding 1.0. With the
    # default step 0.01 that is 0.01..0.99 (99 points). Leaving out the endpoints
    # is deliberate: a threshold of exactly 0.0 calls everything positive and 1.0
    # calls (almost) nothing.
    grid = np.arange(args.grid_step, 1.0, args.grid_step)
    print(f"\nSweeping {len(grid)} thresholds per class...")
    tuned = find_threshold_per_class(L_tr, S_tr, grid)
    for cls, t in zip(classes, tuned):
        print(f"  {cls:6s}  optimal threshold = {t:.2f}")

    # --- Step 5: TEST eval at 0.5 baseline AND at tuned thresholds --------
    print("\nLoading test split...")
    test_ds = _load_split(run_cfg, "test")
    print(f"  test  N = {len(test_ds)}")
    L_te, S_te = run_inference(model, test_ds, device, norm, args.batch_size)

    # base: a flat 0.5 cutoff for every class -> the "before" numbers.
    # tune: the calibrated per-class thresholds -> the "after" numbers.
    # Both go through the SAME scorer, so f1_macro is directly comparable and
    # the AUROC fields are guaranteed equal (threshold-free).
    base = evaluate_at_thresholds(L_te, S_te, [0.5] * len(classes), classes)
    tune = evaluate_at_thresholds(L_te, S_te, tuned, classes)

    print(f"\nTest at 0.5     : Prec={base['precision_macro']:.4f}  "
          f"Rec={base['recall_macro']:.4f}  F1={base['f1_macro']:.4f}  "
          f"AUROC={base['macro_auroc']:.4f}")
    print(f"Test at tuned   : Prec={tune['precision_macro']:.4f}  "
          f"Rec={tune['recall_macro']:.4f}  F1={tune['f1_macro']:.4f}  "
          f"AUROC={tune['macro_auroc']:.4f}")

    # --- Step 6: persist the full result ----------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)   # create results/.../ if absent
    with open(out, "w") as f:
        # indent=2 keeps the JSON diffable in version control / review.
        json.dump({
            "checkpoint":       str(ckpt),
            "dataset_type":     run_cfg.get("dataset_type", "ptbxl"),
            "class_names":      classes,
            "n_train":          int(L_tr.shape[0]),    # N rows actually scored
            "n_test":           int(L_te.shape[0]),
            # class name -> its tuned threshold, a readable top-level view (the
            # same values also live inside tuned.per_class[...]["threshold"]).
            "tuned_thresholds": {cls: round(t, 4) for cls, t in zip(classes, tuned)},
            "baseline_at_0_5":  base,
            "tuned":            tune,
        }, f, indent=2)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
