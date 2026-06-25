"""
scripts/bootstrap_ci.py
=======================
Puts an error bar on the accuracy number. It computes
non-parametric bootstrap confidence intervals on macro AUROC and per-class
AUROC, working either from a saved checkpoint or from a precomputed
(labels, scores) pair written by ensemble_eval.py.

A single AUROC value is a point estimate with no sense of how stable it is;
this script turns it into the '0.XXXX [lo, hi]' interval I quote for PTB-XL and Georgia.
Without the interval, two recipes that differ by 0.003 look meaningfully different
when they are not.

Test-set sampling variance: how much the reported AUROC would move if a
different sample of N test records had been drawn from the same population.

Non-parametric percentile bootstrap (Efron 1979; the simplest variant, and
the choice because AUROC has no clean analytic standard error to
plug into a formula).

for b in 1..B:
    indices_b = uniform random sample of N indices in [0, N) with replacement
    labels_b  = labels[indices_b]
    scores_b  = scores[indices_b]
    auroc_b   = sklearn.roc_auc_score(labels_b, scores_b, average="macro")
report:
    point_estimate = AUROC on the unsampled test set
    CI low         = empirical 2.5 percentile of {auroc_1, ..., auroc_B}
    CI high        = empirical 97.5 percentile of {auroc_1, ..., auroc_B}


Usage
-----
From a saved checkpoint (runs inference once, then bootstraps):
.venv/bin/python -m scripts.bootstrap_ci \\
    --checkpoint results/repro_ptbxl_results_matching/s11/checkpoints/best.pt \\
    --out        results/repro_ptbxl_results_matching/ci_results_matching_s11.json \\
    --bootstraps 1000

From an ensemble_eval output (skips re-inference; reuses the labels.npy +
scores.npy sidecars ensemble_eval.py writes):
.venv/bin/python -m scripts.bootstrap_ci \\
    --labels    results/repro_ptbxl_results_matching/ensemble_labels.npy \\
    --scores    results/repro_ptbxl_results_matching/ensemble_scores.npy \\
    --class-names NORM MI STTC CD HYP \\
    --out       results/repro_ptbxl_results_matching/ci_ensemble_results_matching.json

Output JSON schema
------------------
{
  "method":       "bootstrap-percentile",
  "n_bootstraps": 1000,
  "seed":         0,
  "n_test":       1711,
  "class_names":  ["NORM", "MI", "STTC", "CD", "HYP"],
  "ci": {
    "macro":     {"point": 0.9088, "lo": 0.8990, "hi": 0.9187, "n_valid": 1000},
    "per_class": {
        "NORM": {"point": ..., "lo": ..., "hi": ..., "n_valid": ...},
        ...
    }
  },
  "checkpoint":   "<path>"  (only when --checkpoint mode),
  "dataset_type": "ptbxl" | "georgia"  (only when --checkpoint mode),
  "labels":       "<path>"  (only when --labels/--scores mode),
  "scores":       "<path>"
}

In each block, 'point' is the AUROC computed once on the full (unsampled)
test set, and 'lo'/'hi' are the 2.5/97.5 percentiles of the B-resample
distribution.

Other potential noise sources we do not consider:
- Multi-seed std: training-process noise -- would the AUROC come back the same
  if the model were re-trained from a different random seed on the same data?
  Reported separately as the per-seed mean +/- std printed by sweep harnesses.
- Cross-process variance: the bfloat16 sLSTM kernel sums in a
  scheduling-dependent order, so separate sweep processes drift by
  ~0.005-0.015 AUROC even at a fixed seed. Within one sweep process it is
  deterministic. Also not measured here.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def bootstrap_auroc(labels: np.ndarray, scores: np.ndarray,
                    class_names: list[str], n_boot: int = 1000,
                    seed: int = 0) -> dict:
    """Point estimates plus percentile bootstrap CIs for AUROC.

    Args:
        labels: Multi-hot ground truth, shape (N, C) -- N test records, C
            classes, entries 0/1. One ECG can carry several positive labels at
            once (both the PTB-XL superclasses and the Georgia SNOMED-CT classes
            are multi-label, the way a single tracing can show, say, AF and a
            bundle-branch block together).
        scores: Model probabilities, shape (N, C), lined up row-for-row and
            column-for-column with 'labels'. Post-sigmoid; nothing is
            re-normalized here.
        class_names: Length-C list naming each column in column order. Used as
            the keys of the per-class result and to label the output.
        n_boot: Number of resamples B (default 1000). Cost is O(B * N * C).
        seed: Seed for NumPy's default_rng. Pins the resample sequence so a
            given (labels, scores, n_boot, seed) reproduces exactly.

    Returns:
        dict with two keys:
            "macro": {"point", "lo", "hi", "n_valid"} for the macro-averaged
                AUROC. 'point' is measured once on the full (unsampled) test
                set; 'lo'/'hi' are the 2.5/97.5 percentiles of the resample
                distribution; 'n_valid' is how many of the B resamples gave a
                finite macro AUROC (degenerate ones are dropped).
            "per_class": {class_name -> {"point", "lo", "hi", "n_valid"}}, same
                fields per class. 'point' is None for a class that is
                single-valued on the full test set -- no positives or no
                negatives, so AUROC is undefined for it.
    """
    rng = np.random.default_rng(seed)
    N, C = labels.shape  # N = number of test records, C = number of classes

    # --- Point estimates on the full, unsampled test set --------------------
    macro_point = float(roc_auc_score(labels, scores, average="macro"))
    per_class_point = {}
    for i, cls in enumerate(class_names):
        # AUROC needs both a positive and a negative example to be defined
        # (>=2 distinct label values in the column). A single-valued column ->
        # None, which carries through to the JSON and the printed
        # '(single-class column)'.
        if len(np.unique(labels[:, i])) >= 2:
            per_class_point[cls] = float(roc_auc_score(labels[:, i], scores[:, i]))
        else:
            per_class_point[cls] = None

    # --- Bootstrap loop -----------------------------------------------------
    # macro_samples gathers one macro AUROC per surviving resample;
    # per_class_samples gathers per-class AUROCs keyed by class name. Plain
    # lists rather than preallocated arrays, because degenerate resamples are
    # skipped and the surviving count per class can fall below n_boot.
    macro_samples: list[float] = []
    per_class_samples: dict[str, list[float]] = {cls: [] for cls in class_names}

    for b in range(n_boot):
        # Draw N indices in [0, N) with replacement -- one simulated test set
        # of the same size, the non-parametric resample. Drawing the index
        # vector ONCE and applying it to both labels and scores keeps every
        # class on the SAME simulated patients, so the per-class intervals are
        # comparable and the macro is their mean over those same patients.
        idx = rng.integers(0, N, size=N)

        L_b, S_b = labels[idx], scores[idx]  # both (N, C), gathered by 'idx'

        # If a class comes out all-0 or all-1 in this resample, AUROC is
        # undefined and the macro mean is nan. Keep the macro sample
        # only when finite, so a degenerate resample falls out of the interval
        # instead of poisoning it with nan.
        with warnings.catch_warnings():
            # Mute sklearn's "only one class present" UndefinedMetricWarning:
            # the degeneracy is handled explicitly by the np.isfinite gate
            # below, so the warning would just be console noise B times over.
            warnings.simplefilter("ignore")
            macro_b = roc_auc_score(L_b, S_b, average="macro")
        if np.isfinite(macro_b):
            macro_samples.append(float(macro_b))
        for i, cls in enumerate(class_names):
            # The per-class path drops only the offending class for this
            # iteration, while the macro path above drops the whole iteration.
            # That asymmetry is intentional and is spelled out in the header.
            if len(np.unique(L_b[:, i])) >= 2:
                per_class_samples[cls].append(float(
                    roc_auc_score(L_b[:, i], S_b[:, i])
                ))

        # Heartbeat every 100 iterations so a long B=10000 run shows progress.
        if (b + 1) % 100 == 0:
            print(f"  bootstrap: {b+1}/{n_boot}", flush=True)

    def _ci(samples: list[float]):
        """Reduce a list of bootstrap AUROCs to a percentile-CI summary.

        Args:
            samples: The surviving (finite) AUROC values from the loop.

        Returns:
            {"lo", "hi", "n_valid"} -- the 2.5/97.5 empirical percentiles and
            the sample count. If 'samples' is empty (every resample was
            degenerate for this class), returns lo=hi=None with n_valid=0
            rather than letting np.percentile blow up on an empty array.
        """
        if not samples:
            return {"lo": None, "hi": None, "n_valid": 0}
        # The percentile bootstrap in one line: the empirical 2.5/97.5 quantiles
        # of the resample distribution ARE the 95% interval endpoints -- no
        # normality and no analytic standard error (which AUROC lacks) assumed.
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "n_valid": len(samples)}

    # Glue the point estimate (from the full set) onto the interval endpoints
    # (from the resample distribution) so each metric is one tidy record.
    macro = {"point": macro_point, **_ci(macro_samples)}
    per_class = {
        cls: {"point": per_class_point[cls], **_ci(per_class_samples[cls])}
        for cls in class_names
    }
    return {"macro": macro, "per_class": per_class}


def _from_checkpoint(args):
    """Get (labels, scores) by scoring a saved checkpoint on the test split.

    The '--checkpoint' path: rather than read precomputed sidecar arrays, this
    loads the run config and model from the checkpoint, builds the matching
    test split, and scores it once. The arrays then feed bootstrap_auroc just
    as the '--labels/--scores' path would.

    Args:
        args: Parsed argparse namespace; uses 'args.checkpoint' (path to a
            best.pt) and 'args.batch_size' (inference batch size).

    Returns:
        (labels, scores, class_names, run_cfg) -- labels/scores as (N, C) numpy
        arrays from run_inference, the class-name list, and the full run config
        dict (the caller reads dataset_type out of it for the output metadata).
    """

    from scripts.evaluate_checkpoint import (
        load_run_config, load_model, load_test_dataset, run_inference,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)

    # load_run_config reads the companion result.json the trainer wrote; it
    # carries dataset_type, class_names, normalization, the label rules, etc.
    run_cfg = load_run_config(ckpt_path)
    dataset = load_test_dataset(run_cfg)        # PTB-XL or Georgia test split
    model = load_model(ckpt_path, run_cfg, device)

    labels, scores = run_inference(
        model, dataset, device,
        run_cfg.get("normalization", "zero_mean_unit_var"),
        args.batch_size,
    )
    return labels, scores, run_cfg["class_names"], run_cfg


def main() -> None:
    # CLI entry point: parse args, get (labels, scores), bootstrap, save.

    p = argparse.ArgumentParser()

    # Mutually exclusive and required: exactly one input source must be given;
    # argparse rejects passing both --checkpoint and --labels, or neither.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Path to best.pt; runs inference itself.")
    src.add_argument("--labels",     help="Path to .npy with labels (N, C).")
    # --scores / --class-names only make sense alongside --labels; argparse
    # can't express that conditional requirement, so it is asserted below.
    p.add_argument("--scores",       help="Path to .npy with scores (N, C). "
                                          "Required with --labels.")
    p.add_argument("--class-names",  nargs="+",
                   help="Class names; required with --labels.")
    p.add_argument("--out",          required=True)
    p.add_argument("--bootstraps",   type=int, default=1000)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--batch-size",   type=int, default=512)
    args = p.parse_args()

    if args.checkpoint:
        # Checkpoint mode: get everything (labels, scores, class names) by
        # scoring the test split, and record the checkpoint and dataset in meta.
        labels, scores, class_names, run_cfg = _from_checkpoint(args)
        meta = {"checkpoint": args.checkpoint,
                "dataset_type": run_cfg.get("dataset_type", "ptbxl")}
    else:
        # Sidecar mode: --class-names has to be given explicitly, because the
        # .npy arrays carry no column labels of their own.
        assert args.scores and args.class_names, (
            "--labels requires --scores and --class-names"
        )
        # Cast to the dtypes the bootstrap expects: int32 labels (so np.unique
        # sees clean 0/1) and float32 scores. ensemble_eval.py already writes
        # these, just to be sure.
        labels = np.load(args.labels).astype(np.int32)
        scores = np.load(args.scores).astype(np.float32)
        class_names = args.class_names
        meta = {"labels": args.labels, "scores": args.scores}

    print(f"N={labels.shape[0]}  C={labels.shape[1]}  bootstraps={args.bootstraps}")
    ci = bootstrap_auroc(labels, scores, class_names,
                         n_boot=args.bootstraps, seed=args.seed)

    print(f"\n=== Macro AUROC: "
          f"{ci['macro']['point']:.4f} "
          f"[{ci['macro']['lo']:.4f}, {ci['macro']['hi']:.4f}] ===")
    for cls, c in ci["per_class"].items():
        if c["point"] is None:
            # point is None only when the class was single-valued on the full
            # test set (AUROC undefined); flag it rather than print nan.
            print(f"  {cls:6s}  (single-class column)")
        else:
            print(f"  {cls:6s}  {c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # mkdir -p on the dir

    with open(out_path, "w") as f:
        # '**meta' splices in exactly one of the two provenance blocks
        # (checkpoint+dataset_type OR labels+scores) depending on input mode,
        # so the JSON records where these numbers came from.

        json.dump({
            "method":      "non-parametric bootstrap, 95% percentile CI",
            "n_bootstraps": args.bootstraps,
            "seed":         args.seed,
            "n_test":       int(labels.shape[0]),
            "class_names":  class_names,
            "ci":           ci,
            **meta,
        }, f, indent=2)
    # NOTE: the 95% CI percentiles (2.5 / 97.5) are hard-coded in _ci. Could add alpha as a variable.
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
