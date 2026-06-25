"""
scripts/ensemble_eval.py
========================
Multi-seed prediction ensemble. I average the sigmoid outputs of N
checkpoints (each one a different seed of the same recipe) and report the
same metrics as evaluate_checkpoint.py, but computed on the averaged
scores instead of a single seed.

Usage
-----
Basic (10-seed ensemble of the results_matching PTB-XL recipe):

    .venv/bin/python -m scripts.ensemble_eval \\
        --checkpoints results/repro_ptbxl_results_matching/s*/checkpoints/best.pt \\
        --out         results/repro_ptbxl_results_matching/ensemble.json

Then bootstrap a CI on the ensemble (no re-inference needed):

    .venv/bin/python -m scripts.bootstrap_ci \\
        --labels results/repro_ptbxl_results_matching/ensemble_labels.npy \\
        --scores results/repro_ptbxl_results_matching/ensemble_scores.npy \\
        --class-names NORM MI STTC CD HYP \\
        --out    results/repro_ptbxl_results_matching/ci_ensemble.json

Output JSON schema
------------------

    {
      "ensemble_size":   10,
      "ensemble_auroc":  0.9088,
      "ensemble_metrics": { ... full compute_all_metrics dict ...
                            per_class_auroc, per_class_cm, label_cooccurrence,
                            precision_macro, recall_macro, map ... },
      "per_member": [
        {"checkpoint": "<path>", "test_auroc": 0.9078, "per_class_auroc": {...}},
        ...
      ],
      "checkpoints":     [list of paths],
      "dataset_type":    "ptbxl" | "georgia",
      "class_names":     [...],
      "labels_npy":      "<out>_labels.npy",
      "scores_npy":      "<out>_scores.npy"
    }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_checkpoint import (
    load_run_config,
    load_model,
    load_test_dataset,
    run_inference,
    compute_all_metrics,
)


def main() -> None:
    """CLI entry point: build and evaluate a multi-seed ensemble end to end.

    Args:
        None directly -- arguments come from the command line via argparse:
          --checkpoints  one or more best.pt paths (one per ensemble member).
          --out          output JSON path; the sidecar npy names derive from it.
          --batch-size   inference batch size (default 512), passed through to
                         run_inference.
          --threshold    decision threshold for the count-based metrics
                         (confusion matrices / precision / recall / F1).
    Returns:
        None.

    Raises:
        AssertionError  if any member's dataset_type or class_names disagree
                        with the first member, or if the test-set label order
                        is not identical across members
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to best.pt files for each ensemble member.")
    p.add_argument("--out",         required=True, help="Output JSON path.")
    # batch_size only affects inference speed / memory, not the result.
    p.add_argument("--batch-size",  type=int, default=512)
    # threshold drives the count-based metrics (CM/precision/recall/F1) only;
    # macro AUROC and MAP are computed from the raw scores and ignore it.
    p.add_argument("--threshold",   type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The first checkpoint defines the dataset; later ones must match.
    # I load the test split exactly once here, from the first member's config,
    # and reuse it across every member -- the test records are identical for all
    # seeds of one recipe.
    base_cfg = load_run_config(Path(args.checkpoints[0]))
    dataset = load_test_dataset(base_cfg)

    norm_strategy = base_cfg.get("normalization", "zero_mean_unit_var")

    # Direct index (not .get) on purpose: class_names is mandatory in any modern
    # result.json config, and an ensemble with no known class set is unusable.
    class_names   = base_cfg["class_names"]

    print(f"Ensemble size: {len(args.checkpoints)}")
    print(f"Dataset      : {base_cfg.get('dataset_type', 'ptbxl')}, "
          f"N={len(dataset)}, C={len(class_names)}")

    # Per-member metrics (sanity) and the ensemble accumulator.
    # per_member  -- one record per checkpoint, for the JSON's "per_member" list.
    # sum_scores  -- running sum of the per-seed sigmoid scores; promoted to
    #                float64 on first write so float32 round-off does not
    #                accumulate over the ~10 additions before I divide by N.
    # labels_ref  -- the ground-truth labels from the first member; every later
    #                member must reproduce these exactly (same test order).
    per_member: list[dict] = []
    sum_scores: np.ndarray | None = None
    labels_ref: np.ndarray | None = None

    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt_path = Path(ckpt_path)
        run_cfg   = load_run_config(ckpt_path)

        # Cross-checkpoint sanity -- refuse to ensemble mismatched runs.
        # Averaging scores across a PTB-XL run and a Georgia run (different class
        # sets, different test sets).
        assert run_cfg.get("dataset_type") == base_cfg.get("dataset_type"), (
            f"Member {i} dataset_type mismatch: "
            f"{run_cfg.get('dataset_type')!r} vs {base_cfg.get('dataset_type')!r}"
        )
        assert run_cfg["class_names"] == class_names, (
            f"Member {i} class_names mismatch."
        )

        print(f"\n[{i+1}/{len(args.checkpoints)}] {ckpt_path}")
        model = load_model(ckpt_path, run_cfg, device)

        # labels and scores are both (N, C): labels are the int32 ground truth,
        # scores the float32 probabilities (forward() already applies the
        # sigmoid). The eval pipeline (reshape -> normalize -> nan_to_num) lives
        # inside run_inference and is shared with train.py's evaluate().
        labels, scores = run_inference(model, dataset, device,
                                        norm_strategy, args.batch_size)

        if labels_ref is None:
            # First member: seed the accumulator and pin the label reference.
            labels_ref = labels
            sum_scores = scores.astype(np.float64)
        else:
            # Every later member must see the test set in the exact same order;
            # otherwise sum_scores[k] and labels_ref[k] describe different
            # records and the average is wrong. The loader uses
            # shuffle=False, so this always holds -- the assert is a
            # tripwire against a future non-deterministic dataset change.
            assert (labels_ref == labels).all(), (
                "Test-set label order changed between members; the dataset "
                "loader must be deterministic with shuffle=False."
            )
            # Accumulate in float64 (scores arrive float32).
            sum_scores += scores.astype(np.float64)

        # Per-seed metrics on this member's own (un-averaged) scores. I only
        # surface the macro and per-class AUROC in the JSON -- enough to spot an
        # outlier seed or a broken checkpoint -- but compute_all_metrics returns
        # the full battery, so the threshold still has to be passed through.
        member_metrics = compute_all_metrics(
            labels, scores, class_names, args.threshold
        )
        per_member.append({
            "checkpoint": str(ckpt_path),
            "test_auroc": member_metrics["macro_auroc"],
            "per_class_auroc": member_metrics["per_class_auroc"],
        })
        print(f"  member auroc = {member_metrics['macro_auroc']:.4f}")

        # Free GPU memory before loading the next checkpoint.
        # Each member is a fresh model; without dropping the previous one the
        # peak VRAM would grow ~linearly with ensemble size. del drops the last
        # Python reference; empty_cache returns the freed blocks to the driver
        # so the next load_model starts from a clean allocator.
        # NOTE: empty_cache is a no-op on CPU, so this is safe in CPU mode too.
        del model
        torch.cuda.empty_cache()

    # Equal-weight average of the per-seed sigmoid outputs (the 1/N ensemble).
    # Cast back to float32 for the metric battery and the on-disk sidecar -- the
    # extra float64 precision was only needed while summing.
    mean_scores = (sum_scores / len(args.checkpoints)).astype(np.float32)

    ensemble_metrics = compute_all_metrics(
        labels_ref, mean_scores, class_names, args.threshold
    )

    print(f"\n=== ENSEMBLE  AUROC = {ensemble_metrics['macro_auroc']:.4f} ===")
    # Per-class AUROC is a dict keyed by class name; print one line each, padded
    # to 6 chars so the column stays aligned for both the 5-class PTB-XL and the
    # 7-class Georgia label sets.
    for cls, v in ensemble_metrics["per_class_auroc"].items():
        print(f"  {cls:6s}  {v:.4f}")

    out_path = Path(args.out)
    # Create the output directory tree if the caller pointed --out at a fresh
    # results/ subdir; exist_ok keeps re-runs idempotent.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sidecar .npy files let scripts/bootstrap_ci.py compute CIs on the
    # ensemble without re-running inference.
    # Naming: strip any suffix from --out and append "_labels.npy"/"_scores.npy"
    # to the stem, so foo.json yields foo_labels.npy and foo_scores.npy next to
    # it. The two paths are recorded in the JSON so bootstrap_ci can be pointed
    # straight at them.
    labels_npy = out_path.with_suffix("").with_name(out_path.stem + "_labels.npy")
    scores_npy = out_path.with_suffix("").with_name(out_path.stem + "_scores.npy")

    # Labels saved as int32 (0/1 multi-hot) and scores as the float32 averaged
    # probabilities -- exactly the (N, C) shapes bootstrap_ci.py expects from its
    # --labels/--scores entry point.
    np.save(labels_npy, labels_ref.astype(np.int32))
    np.save(scores_npy, mean_scores)

    # Top-level JSON schema is documented in the module docstring; keep these
    # keys stable since downstream report tooling reads ensemble_auroc and the
    # *_npy paths by name.
    with open(out_path, "w") as f:
        json.dump({
            "ensemble_size":  len(args.checkpoints),
            "ensemble_auroc": ensemble_metrics["macro_auroc"],
            "ensemble_metrics": ensemble_metrics,
            "per_member":     per_member,
            # Store the raw CLI strings (not the Path objects) for provenance --
            # this is the exact list that produced these numbers.
            "checkpoints":    [str(p) for p in args.checkpoints],
            # Default to "ptbxl" to match evaluate_checkpoint.py's fallback for
            # pre-result.json checkpoints that lack an explicit dataset_type.
            "dataset_type":   base_cfg.get("dataset_type", "ptbxl"),
            "class_names":    class_names,
            "labels_npy":     str(labels_npy),
            "scores_npy":     str(scores_npy),
        }, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {labels_npy}")
    print(f"Wrote {scores_npy}")


if __name__ == "__main__":
    main()
