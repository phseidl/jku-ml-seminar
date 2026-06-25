"""
src/training/train.py
=====================
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json, math
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, OneCycleLR,
    ReduceLROnPlateau, SequentialLR, StepLR,
)
from src.data.dataset import PTBXLDataset
from src.models.xlstm_ecg import XLSTMECGModel
from pathlib import Path


def apply_random_masking(x: torch.Tensor, mask_ratio: float, mask_prob: float) -> torch.Tensor:
    """
    Random masking augmentation (Section 3.7 of the original study).

    This blanks out short stretches of the ECG in time so the model can't
    lean on any one segment to make its call -- a defense against overfitting,
    in the spirit of dropout but applied to the signal itself. Used at train
    time only (see train_epoch).

    Parameters
    ----------
    x : torch.Tensor
        Already-reshaped, already-normalized batch of shape
        (B, seq_len=T, feat_dim=12*F). I mask whole time frames (rows of the
        T axis): all 12*F features at a chosen time step go to zero together.
    mask_ratio : float
        Fraction of time frames masked in a chosen ECG (original study: 0.2).
    mask_prob : float
        Per-ECG probability that masking is applied at all (original study:
        0.8); so 20% of a batch passes through untouched.

    Returns
    -------
    torch.Tensor
        Same shape as x.
    """
    batch_size, seq_len, feat_dim = x.shape   # feat_dim is bound but unused; kept for shape clarity

    x = x.clone()
    for i in range(batch_size):

        # Independent Bernoulli(mask_prob) draw per ECG -- some in the batch
        # are masked, some are not, which is the intended regularization.
        if torch.rand(1).item() < mask_prob:

            # max(1, ...) guarantees at least one frame is masked even when
            # seq_len * mask_ratio rounds down to 0 (e.g. tiny seq_len), so the
            # augmentation is never a silent no-op on a chosen ECG.
            # At seq_len=59 this never binds (0.2*59=11); it only matters for very
            # short sequences, so it does not affect the reported numbers.
            num_masked  = max(1, int(seq_len * mask_ratio))

            # randperm(seq_len)[:k] = sample k DISTINCT frame indices without
            # replacement, so we never double-count a frame toward the ratio.
            mask_indices = torch.randperm(seq_len)[:num_masked]

            # Zero the whole feature vector at each chosen time step.
            x[i, mask_indices, :] = 0.0

    return x


def train_epoch(model, loader, optimizer, criterion, device, cfg):
    """
    Run one full pass over the training DataLoader and return the mean loss.

    Parameters
    ----------
    model : nn.Module
        The XLSTMECGModel; put in train() mode here (dropout active).
    loader : DataLoader
        Yields (x, labels) where x is (B, 12, 241, 59) STFT tensors and labels
        is (B, C) multi-hot. Should be the shuffled TRAIN loader.
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss operating on sigmoid probs.
    device : torch.device
        Where tensors and model live.
    cfg : DictConfig
        Reads cfg.training.normalization, cfg.training.grad_clip_max_norm,
        cfg.model.mask_prob, cfg.model.mask_ratio.

    Returns
    -------
    float
        Mean per-batch loss over the epoch (sum of loss.item() / num batches).

    """
    model.train()                  # enable dropout
    total_loss    = 0.0

    norm_strategy = getattr(cfg.training, "normalization", "zero_mean_unit_var")
    grad_clip     = float(getattr(cfg.training, "grad_clip_max_norm", 1.0))

    # Fallbacks match the values from the original study (mask_prob 0.8,
    # mask_ratio 0.2) and config.yaml, so a config that omits these keys trains
    # with that masking rather than a divergent default.
    mask_prob     = float(getattr(cfg.model, "mask_prob", 0.8))
    mask_ratio    = float(getattr(cfg.model, "mask_ratio", 0.2))

    for x, labels in loader:
        x      = x.to(device)          # (B, 12, 241, 59) = (batch, leads, freq, time)

        # labels arrive as int/bool multi-hot; BCE-family losses need float
        # targets in [0, 1], hence the .float() cast.
        labels = labels.float().to(device)

        # Reshape to (B, seq=59, features=2892).
        # permute (0, 3, 1, 2): (B, 12, 241, 59) -> (B, 59, 12, 241), moving the
        # TIME axis (59) to position 1 so it becomes the sequence axis the xLSTM
        # consumes. reshape(B, 59, -1) then flattens (12 leads x 241 freq) = 2892
        # into the feature axis. The model reads each of the 59 time frames as
        # one token carrying a 2892-dim feature vector -- the ECG turned into a
        # short sequence of spectral snapshots.

        # Get batch size from length.
        B = x.size(0)

        x = x.permute(0, 3, 1, 2).reshape(B, 59, -1)

        x = normalize_batch(x, norm_strategy)

        # nan_to_num is REQUIRED, not defensive padding: the bfloat16 sLSTM CUDA
        # kernel can emit NaN/Inf on degenerate (near-constant) inputs, and a
        # single NaN propagates through .backward() and poisons every parameter.
        # Clamping to 0 here keeps one bad batch from killing the whole run.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Augmentation is the LAST input transform, after normalization, so the
        # zeroed frames stay exactly zero (a masked frame must read as "absent",
        # not as a normalized value). Guarded by mask_prob > 0 so a config can
        # turn augmentation off entirely without paying the clone cost.
        if mask_prob > 0.0:
            x = apply_random_masking(x, mask_ratio, mask_prob)

        optimizer.zero_grad()

        preds = model(x)               # (B, C) sigmoid probabilities

        loss  = criterion(preds, labels)
        loss.backward()

        # Global-norm gradient clipping to keep training stable. grad_clip <= 0
        # disables it; the default 1.0 follows the original study / config.yaml.
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # .item() pulls the scalar to host and detaches it from the graph, so we
        # accumulate a plain float and don't retain the batch's autograd graph.
        total_loss += loss.item()

    # Mean over batches (len(loader) = number of batches), so this
    # is a per-batch mean and is only comparable across epochs at a fixed batch
    # size.
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
        model:     nn.Module,
        loader:    DataLoader,
        criterion: nn.Module,
        device:    torch.device,
        cfg:       DictConfig,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluate the model on a full val/test DataLoader (no gradient, no masking).

    Decorated with @torch.no_grad() so the whole pass runs without building an
    autograd graph.
    Alternative: torch.no_grad().

    Parameters
    ----------
    model : nn.Module
        Put in eval() mode here (dropout off, deterministic forward).
    loader : DataLoader
        The val or test loader (shuffle=False); yields (x, y) batches with the
        same (B, 12, 241, 59) / (B, C) shapes as the train loader.
    criterion : nn.Module
        Same loss object used in training; its value is reported for monitoring.
    device : torch.device
    cfg : DictConfig
        Reads cfg.training.normalization ONLY -- and it MUST be the value used
        in train_epoch, or eval preprocesses the ECG differently from training.

    Returns
    -------
    (mean_loss, all_predictions, all_labels) : tuple[float, Tensor, Tensor]
        mean_loss        : float, per-batch mean of the criterion
        all_predictions  : (N, C) sigmoid probabilities for the WHOLE split,
                           on CPU, concatenated in loader order
        all_labels       : (N, C) multi-hot ground truth, on CPU
        N = total ECGs in the split, C = num classes. These two arrays feed
        straight into compute_metrics / compute_label_cooccurrence.
    """
    model.eval()                   # disable dropout for a deterministic pass
    total_loss    = 0.0
    all_preds     = []             # list of per-batch (B, C) prediction tensors
    all_labels    = []             # list of per-batch (B, C) label tensors
    norm_strategy = getattr(cfg.training, "normalization", "zero_mean_unit_var")

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        # Same reshape as train_epoch, written out symbolically here:
        # (B, n_leads=12, n_freq=241, n_frames=59) -> permute time to axis 1 ->
        # (B, 59, 12*241=2892). Reading the dims from x.shape (rather than the
        # literal 59/2892) means a config with a different nfft/lead count still
        # reshapes correctly without code changes.
        B, n_leads, n_freq, n_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, n_frames, n_leads * n_freq)

        x = normalize_batch(x, norm_strategy)

        # Same bfloat16-kernel NaN guard as training.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # No apply_random_masking here on purpose: the metrics have to reflect
        # how the model behaves on the intact ECG.
        preds = model(x)
        loss  = criterion(preds, y)
        total_loss += loss.item()

        # Move to CPU as we go so GPU memory holds only one batch's activations
        # at a time; the full (N, C) arrays can be large for the test split.
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    # Concatenate the per-batch chunks into split-wide (N, C) arrays.
    all_preds  = torch.cat(all_preds,  dim=0)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N, C)

    return total_loss / len(loader), all_preds, all_labels


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor,
                    threshold: float = 0.5,
                    class_names: list[str] | None = None) -> dict:
    """
    Compute the full metric set for multi-label ECG classification.
    """

    # Local imports: numpy + sklearn are only needed for metrics, not for the
    # training hot loop, so we keep them out of the module-level import cost.
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    C = preds.shape[1]             # number of classes = width of the label/score matrix
    if class_names is None:
        # Generic fallback names so per-class dicts are still keyed sensibly
        # when a caller doesn't supply real class labels.
        class_names = [f"C{i}" for i in range(C)]

    # Accept either torch tensors (have .numpy()) or array-likes. hasattr guards
    # the duck-typing so this helper works from a script that already holds
    # numpy arrays (e.g. the ensemble path) without an explicit tensor.
    scores   = preds.numpy() if hasattr(preds,   "numpy") else np.array(preds)
    labels_np = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)
    labels_int = labels_np.astype(int)            # ground truth as 0/1 ints

    # Threshold the continuous scores into hard 0/1 calls for the confusion-
    # matrix / precision-recall block -- this is the model committing to a
    # diagnosis. AUROC and MAP below ignore the threshold and rank the raw
    # scores instead.
    preds_bin  = (scores >= threshold).astype(int)

    # -- Macro AUROC -----------------------------------------------------------
    # Mean of the per-class AUROCs over classes that are well-defined in this
    # split (both labels present). Equivalent to sklearn average="macro" when
    # every class is present (the test-set case), but robust to a degenerate
    # column, which sklearn would otherwise turn into a nan macro.
    # The "if len(np.unique(...)) >= 2" filter skips any class that is all-0 or
    # all-1 in this split: AUROC is undefined for a single-class column and
    # sklearn would raise / return nan, dragging the macro mean to nan. We drop
    # such columns from the mean instead. On the full test set every class is
    # present, so this collapses to the standard macro AUROC.
    _auroc_defined = [
        float(roc_auc_score(labels_int[:, i], scores[:, i]))
        for i in range(C) if len(np.unique(labels_int[:, i])) >= 2
    ]

    # Empty list -> 0.0 rather than nan, so a degenerate early-epoch val pass
    # can't crash the best-checkpoint comparison in train().
    macro_auroc = float(np.mean(_auroc_defined)) if _auroc_defined else 0.0

    # -- Overall element-wise accuracy (eq. 17 of the original study) ---------
    # Mean over EVERY (ECG, class) cell of the multi-hot matrix, not a per-ECG
    # exact-match. Because each ECG carries only a few of the labels, the matrix
    # is mostly true negatives, so this number sits high and flatters the model
    # -- it is kept only to match eq. 17 of the original study for parity. AUROC
    # is the metric I actually judge the model on.
    overall_acc = float((labels_int == preds_bin).mean())

    # -- Per-class AUROC -------------------------------------------------------
    # Same degenerate-column guard as the macro AUROC, but here a skipped class
    # is recorded as None (rather than dropped) so the per-class dict always has
    # one entry per class -- downstream plotting code keys on class name.
    per_class_auroc = {}
    for i, cls in enumerate(class_names):
        if len(np.unique(labels_int[:, i])) >= 2:
            per_class_auroc[cls] = round(
                float(roc_auc_score(labels_int[:, i], scores[:, i])), 4
            )
        else:
            per_class_auroc[cls] = None

    # -- Per-class confusion matrix + derived metrics --------------------------
    # The clinically meaningful view.
    per_class_cm = {}
    all_sen, all_spe, all_pre, all_f1 = [], [], [], []
    for i, cls in enumerate(class_names):
        yt, yp = labels_int[:, i], preds_bin[:, i]   # this class's true / predicted columns
        # Boolean masks combined with & give the four confusion-matrix counts.
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        # Each ratio guards its own denominator: a class with no positives
        # (tp+fn==0) yields sensitivity 0.0 rather than a ZeroDivisionError.
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = recall = how many true cases were caught
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # how many healthy ECGs were left alone
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # of the flagged ECGs, how many were real


        acc = (tp + tn) / (tp + tn + fp + fn)

        # F1 = harmonic mean of precision and recall; guarded so pre==sen==0
        # gives 0.0 instead of 0/0.
        f1  = 2 * pre * sen / (pre + sen) if (pre + sen) > 0 else 0.0
        per_class_cm[cls] = {
            "TP":          tp,
            "TN":          tn,
            "FP":          fp,
            "FN":          fn,
            "sensitivity": round(sen, 4),   # recall
            "specificity": round(spe, 4),
            "precision":   round(pre, 4),
            "accuracy":    round(acc, 4),
            "f1":          round(f1,  4),
        }
        # Collect per-class values to average into the macro figures below.
        all_sen.append(sen); all_spe.append(spe)
        all_pre.append(pre); all_f1.append(f1)

    # -- Macro Precision / Recall / F1 (for Georgia Table 7) ------------------
    # Unweighted mean over classes (macro), so a rare arrhythmia counts as much
    # as a common one -- the right view when the rare class is the dangerous
    # one. recall_macro reuses all_sen because sensitivity == recall.
    # NOTE: this is the mean of the per-class F1s, NOT an F1 recomputed from
    # macro-P and macro-R; the two differ, and Table 7 reports this
    # per-class-then-average form.
    precision_macro = float(np.mean(all_pre))
    recall_macro    = float(np.mean(all_sen))
    f1_macro        = float(np.mean(all_f1))

    # -- MAP: mean average precision, threshold-free (for Georgia Table 7) ----
    # Mean of per-class average precision over classes with at least one
    # positive; equals sklearn average="macro" when every class has positives
    # (the test-set case), but robust to an empty column.
    # "sum() > 0" keeps only classes with at least one positive: average
    # precision is undefined for an all-negative column. AP only
    # needs a positive, since it ranks the positives against everything else.
    _ap_defined = [
        float(average_precision_score(labels_int[:, i], scores[:, i]))
        for i in range(C) if labels_int[:, i].sum() > 0
    ]
    map_score = float(np.mean(_ap_defined)) if _ap_defined else 0.0

    return {
        # Primary keys (unchanged; sweep scripts and evaluator read these)
        "auroc":            round(macro_auroc,    4),
        "accuracy":         round(overall_acc,    4),
        # Extended per-class breakdown
        "per_class_auroc":  per_class_auroc,
        "per_class_cm":     per_class_cm,
        # Aggregate thresholded metrics (Georgia Table 7)
        "precision_macro":  round(precision_macro, 4),
        "recall_macro":     round(recall_macro,    4),
        "f1_macro":         round(f1_macro,        4),
        "map":              round(map_score,       4),
    }


def compute_label_cooccurrence(labels: torch.Tensor, preds: torch.Tensor,
                                class_names: list[str],
                                threshold: float = 0.5) -> dict:
    """
    Count how often pairs of diagnoses appear together -- the label
    co-occurrence matrices for Figures 5 / 6 of the report.

    Clinically this matters because ECG findings travel in company (an old
    infarct often comes with conduction disease, say), and a model can learn to
    confuse two labels that almost always co-occur. The true-vs-predicted matrix
    makes that confusion visible.

    Parameters
    ----------
    labels : Tensor | array, shape (N, C)
        Multi-hot ground truth.
    preds : Tensor | array, shape (N, C)
        Sigmoid scores; binarized at threshold for the predicted matrix.
    class_names : list[str]
        Length C; defines matrix row/column order and is echoed in the result.
    threshold : float
        Cutoff applied to preds before counting (default 0.5).

    Returns
    -------
    dict with:
        true_vs_true      C x C ground-truth self-cooccurrence (Figure 5a / 6a):
                          entry [i, j] = number of ECGs labeled BOTH class i and
                          class j (diagonal = per-class positive count).
        true_vs_predicted C x C cross-cooccurrence (Figure 5b / 6b):
                          entry [i, j] = ECGs with TRUE class i and PREDICTED
                          class j -- a confusion-style co-occurrence, so this
                          matrix is NOT symmetric.
        class_names       the row/column order, for the plotter.
    """
    import numpy as np
    C = len(class_names)

    # Duck-typed numpy coercion, mirroring compute_metrics. Predictions are
    # thresholded to 0/1 here; the true matrix uses the labels as-is.
    ln = labels.numpy().astype(int) if hasattr(labels, "numpy") else np.array(labels, dtype=int)
    pn = (preds.numpy() >= threshold).astype(int) if hasattr(preds, "numpy") else (np.array(preds) >= threshold).astype(int)

    def _cooccur(a, b):
        # Count ECGs where a's class-i bit AND b's class-j bit are both set.
        # O(C^2) cells, each a vectorized AND-and-sum over N ECGs. C is small
        # (5 for PTB-XL, 7 for Georgia) so the double loop is fine.
        mat = np.zeros((C, C), dtype=int)
        for i in range(C):
            for j in range(C):
                mat[i, j] = int(np.sum((a[:, i] == 1) & (b[:, j] == 1)))
        return mat.tolist()    # plain nested lists so the dict is JSON-serializable

    return {
        "class_names":       class_names,
        "true_vs_true":      _cooccur(ln, ln),   # symmetric, diagonal = class counts
        "true_vs_predicted": _cooccur(ln, pn),   # asymmetric: rows=true, cols=predicted
    }


def train(cfg):
    """
    Parameters
    ----------
    cfg : DictConfig
        The fully-merged OmegaConf config (config.yaml + CLI overrides).

    Returns
    -------
    None. All outputs are side effects: checkpoints under
    cfg.training.checkpoint_dir, a rolling train_log.json under
    cfg.training.log_dir, and (optionally) the result.json at
    cfg.training.result_file. Sweep harnesses read those files, not a return
    value, because each run is a separate subprocess.
    """

    # -- Reproducibility seeding ----------------------------------------------
    # Seed every RNG the run touches -- Python's random, numpy, torch on CPU,
    # and all CUDA devices -- so a repeated run shuffles, masks, and initializes
    # the same way. Caveat from the module header still applies: the bfloat16
    # sLSTM kernel makes results deterministic only within a process, not
    # bit-identical across separate processes.
    seed = int(getattr(cfg.training, "seed", 42))

    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device       = torch.device(cfg.training.device)

    # dataset_type is the top-level branch: it picks the Dataset class, the
    # class_names, and which cfg.data sub-keys are read below.
    dataset_type = getattr(cfg.data, "dataset_type", "ptbxl")

    # -- Dataset loading -------------------------------------------------------
    if dataset_type == "ptbxl":

        # ORDER IS IMPORTANT: this list defines column index -> class name for
        # every per-class metric and matches PTBXLDataset.SUPERCLASSES and the
        # index map in evaluate_checkpoint.py (0 NORM normal, 1 MI myocardial
        # infarction, 2 STTC ST/T change, 3 CD conduction disturbance,
        # 4 HYP hypertrophy). Reordering here silently mislabels every figure.
        class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

        # label_aggregation controls how the raw scp_codes -> 5-class labels are
        # derived; "lik_eq_100" keeps only diagnoses the cardiologist marked
        # at 100% confidence.
        ptbxl_agg = getattr(cfg.data, "label_aggregation", "lik_eq_100")

        # nfft drives the STFT frequency resolution and therefore
        # model.input_size. Default 480 was the best nfft in Table 2 of the
        # original study. Sweeping nfft needs a separate cache_dir per value
        # (cache files have shape (12, F, T) with F = nfft//2+1).
        ptbxl_nfft = int(getattr(cfg.data, "nfft", 480))

        # The STFT yields F = nfft//2 + 1 frequency bins per lead, so the model's
        # per-time-step feature width is 12 leads x F. We DERIVE it from nfft and
        # overwrite cfg.model.input_size before the model is built, so a sweep
        # that changes nfft can't get a mismatched input projection from a stale config value.
        derived_input_size = 12 * (ptbxl_nfft // 2 + 1)

        if int(cfg.model.input_size) != derived_input_size:
            print(f"[train] overriding cfg.model.input_size {cfg.model.input_size} "
                  f"-> {derived_input_size} for nfft={ptbxl_nfft}", flush=True)
            cfg.model.input_size = derived_input_size
        train_ds = PTBXLDataset(cfg.data.data_dir, split="train",
                                cache_dir=cfg.data.cache_dir,
                                label_aggregation=ptbxl_agg, nfft=ptbxl_nfft)
        val_ds   = PTBXLDataset(cfg.data.data_dir, split="val",
                                cache_dir=cfg.data.cache_dir,
                                label_aggregation=ptbxl_agg, nfft=ptbxl_nfft)
        test_ds  = PTBXLDataset(cfg.data.data_dir, split="test",
                                cache_dir=cfg.data.cache_dir,
                                label_aggregation=ptbxl_agg, nfft=ptbxl_nfft)

    elif dataset_type == "georgia":
        from src.data.georgia_dataset import GeorgiaECGDataset, GEORGIA_CLASSES
        class_names    = GEORGIA_CLASSES
        georgia_dir    = cfg.data.georgia_dir
        georgia_cache  = getattr(cfg.data, "georgia_cache_dir", None)
        # split_strategy: "default" reproduces our 3-way split (g3-g11/g2/g1);
        # "paper_strict" matches Section 4.7 of the original study exactly
        # (g2-g11 train, g1 test, val == train).
        split_strategy = getattr(cfg.data, "georgia_split_strategy", "default")

        # drop_no_target_codes: True (default, matches all prior reproductions)
        # drops the 40.8% of Georgia records carrying none of the 7 target
        # SNOMED codes -- ECGs whose diagnosis is simply outside the label set we
        # study. False instead keeps them as all-zero label vectors.
        drop_no_target = bool(getattr(cfg.data, "georgia_drop_no_target_codes", True))

        train_ds = GeorgiaECGDataset(georgia_dir, split="train",
                                     cache_dir=georgia_cache,
                                     split_strategy=split_strategy,
                                     drop_no_target_codes=drop_no_target)
        val_ds   = GeorgiaECGDataset(georgia_dir, split="val",
                                     cache_dir=georgia_cache,
                                     split_strategy=split_strategy,
                                     drop_no_target_codes=drop_no_target)
        test_ds  = GeorgiaECGDataset(georgia_dir, split="test",
                                     cache_dir=georgia_cache,
                                     split_strategy=split_strategy,
                                     drop_no_target_codes=drop_no_target)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    # Shared loader kwargs. pin_memory=True speeds the host->GPU copy; only the
    # train loader shuffles (val/test must stay in a fixed order so the (N, C)
    # arrays line up across epochs and across re-evaluation runs).
    dl_kw = dict(batch_size=cfg.data.batch_size,
                 num_workers=cfg.data.num_workers,
                 pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = XLSTMECGModel(cfg).to(device)

    # build_loss_fn takes train_loader because the bce_weighted variant scans it
    # once to count positives per class for pos_weight. The other losses ignore
    # the loader but the signature stays uniform.
    criterion = build_loss_fn(cfg, train_loader, device)

    optimizer = build_optimizer(cfg, model)

    # build_scheduler returns (scheduler, mode); mode tells the loop below
    # whether to call scheduler.step() or scheduler.step(val_auroc).
    scheduler, sched_mode = build_scheduler(cfg, optimizer, len(train_loader))

    # mkdir(parents=True, exist_ok=True): create the output dirs if missing,
    # no-op if they already exist (a sweep may reuse a parent).
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    log_dir  = Path(cfg.training.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Early-stopping / best-checkpoint bookkeeping.
    best_val_auroc = 0.0       # highest macro val AUROC seen so far
    best_epoch     = 0         # epoch that achieved it (for the log/result)
    no_improve     = 0         # epochs since the last improvement
    log_entries    = []        # per-epoch dicts, flushed to train_log.json
    t_start        = time.time()

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion, device, cfg)
        val_loss, val_preds, val_labels = evaluate(model, val_loader,
                                                    criterion, device, cfg)
        val_metrics = compute_metrics(val_preds, val_labels,
                                      class_names=class_names)
        val_auroc   = val_metrics["auroc"]   # macro AUROC = the model-selection metric

        # Scheduler dispatch: ReduceLROnPlateau needs the metric it watches
        # (mode="max" on val AUROC); all the others step blindly per epoch.
        if sched_mode == "plateau":
            scheduler.step(val_auroc)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        log_entries.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "val_auroc":  round(val_auroc,  4),
            "lr":         lr,
        })
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  auroc={val_auroc:.4f}  lr={lr:.2e}", flush=True)

        # Strict ">" so ties do NOT overwrite the best checkpoint or reset the
        # patience counter -- the earliest epoch reaching a given AUROC wins.
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch     = epoch
            no_improve     = 0
            # best.pt is the ONLY checkpoint the final test eval reloads. The
            # optimizer state is saved too so a run could be resumed, though the
            # current pipeline does not resume.
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_auroc": val_auroc},
                       ckpt_dir / "best.pt")
        else:
            no_improve += 1

        # Periodic snapshot every 10th epoch for post-hoc inspection; these are
        # never loaded by train() itself (the final eval uses best.pt only).
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / f"epoch_{epoch:04d}.pt")

        # Rewrite the full log each epoch so a crashed
        # run still leaves a complete log up to the last finished epoch.
        with open(log_dir / "train_log.json", "w") as f:
            json.dump(log_entries, f, indent=2)

        # Early stop once we've gone patience epochs without a new best. Under
        # the Georgia paper_strict split (patience=999) this effectively never
        # fires, which is intentional.
        if no_improve >= cfg.training.early_stopping_patience:
            print(f"Early stop at epoch {epoch} "
                  f"(best val AUROC={best_val_auroc:.4f} @ epoch {best_epoch})")
            break

    # -- Final test evaluation on best checkpoint ------------------------------
    # Reload best.pt before the test eval so the headline number reflects the
    # best-generalizing epoch, NOT whatever weights the loop happened to end on
    # (which after early stopping are patience epochs past the best).
    # map_location=device handles a ckpt saved on a different device.
    model.load_state_dict(
        torch.load(ckpt_dir / "best.pt", map_location=device)["model"])
    test_loss, test_preds, test_labels = evaluate(model, test_loader,
                                                   criterion, device, cfg)
    test_metrics = compute_metrics(test_preds, test_labels,
                                   class_names=class_names)
    test_auroc   = test_metrics["auroc"]
    test_acc     = test_metrics["accuracy"]
    elapsed      = time.time() - t_start

    # Label co-occurrence matrices.
    label_cooccurrence = compute_label_cooccurrence(
        test_labels, test_preds, class_names
    )

    print(f"\n=== TEST  AUROC={test_auroc:.4f}  acc={test_acc:.4f}"
          f"  best_val={best_val_auroc:.4f}@{best_epoch}  {elapsed/60:.1f}min ===")
    print(f"    Prec={test_metrics['precision_macro']:.4f}"
          f"  Rec={test_metrics['recall_macro']:.4f}"
          f"  F1={test_metrics['f1_macro']:.4f}"
          f"  MAP={test_metrics['map']:.4f}")
    print(f"    Per-class AUROC: {test_metrics['per_class_auroc']}")

    # result.json is written ONLY when result_file is set: a sweep sets it per run; a standalone
    # run_training leaves it empty and writes nothing here.
    result_file = getattr(cfg.training, "result_file", "")
    if result_file:
        Path(result_file).parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump({
                # Primary metrics (unchanged keys; sweep scripts read these).
                # round(..., 6): trim float noise but keep enough precision to
                # distinguish near-tied configs in a sweep table.
                "test_auroc":         round(test_auroc, 6),
                "test_accuracy":      round(test_acc,   6),
                "best_val_auroc":     round(best_val_auroc, 6),
                "best_epoch":         best_epoch,
                # epoch is the loop variable, so after an early-stop break this
                # is the epoch we actually stopped at (not num_epochs).
                "total_epochs":       epoch,
                "elapsed_sec":        round(elapsed, 1),
                # Extended metrics (Georgia Table 7 + PTB-XL Figures 4, 5)
                "precision_macro":    test_metrics["precision_macro"],
                "recall_macro":       test_metrics["recall_macro"],
                "f1_macro":           test_metrics["f1_macro"],
                "map":                test_metrics["map"],
                "per_class_auroc":    test_metrics["per_class_auroc"],
                "per_class_cm":       test_metrics["per_class_cm"],
                "label_cooccurrence": label_cooccurrence,
                # Config snapshot for evaluate_checkpoint.py to read back. This
                # records the exact knobs the run used so a re-evaluation
                # rebuilds an identical model and identical dataset splits.
                # Cross-dataset keys are set to None for the inapplicable dataset
                # (e.g. label_aggregation is None on a Georgia run) so a reader
                # can tell which branch produced the file.
                "config": {
                    "dataset_type":  dataset_type,
                    "class_names":   class_names,
                    "normalization": getattr(cfg.training, "normalization",
                                            "zero_mean_unit_var"),
                    "embedding_dim": cfg.model.embedding_dim,
                    "num_blocks":    cfg.model.num_blocks,
                    "num_heads":     cfg.model.num_heads,
                    "dropout":       cfg.model.dropout,
                    "pooling":       getattr(cfg.model, "pooling", "mean"),
                    "fusion_type":   getattr(cfg.model, "fusion_type", "layer"),
                    "slstm_backend": cfg.model.slstm_backend,
                    "input_size":    cfg.model.input_size,
                    "num_classes":   cfg.model.num_classes,
                    "data_dir":      (cfg.data.georgia_dir
                                      if dataset_type == "georgia"
                                      else cfg.data.data_dir),
                    "cache_dir":     (getattr(cfg.data, "georgia_cache_dir", None)
                                      if dataset_type == "georgia"
                                      else cfg.data.cache_dir),
                    "georgia_split_strategy": getattr(
                        cfg.data, "georgia_split_strategy", "default"
                    ) if dataset_type == "georgia" else None,
                    "georgia_drop_no_target_codes": bool(getattr(
                        cfg.data, "georgia_drop_no_target_codes", True
                    )) if dataset_type == "georgia" else None,
                    "label_aggregation": getattr(
                        cfg.data, "label_aggregation", "lik_eq_100"
                    ) if dataset_type == "ptbxl" else None,
                    "nfft": int(getattr(cfg.data, "nfft", 480)) if dataset_type == "ptbxl" else None,
                },
            }, f, indent=2)


# -- Loss functions ---------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss operating on sigmoid probabilities (not logits).

    Turns down the loss on the easy, already-correct calls so the gradient
    spends itself on the hard and rare ones instead -- the rare classes are
    usually the clinically important arrhythmias, and a plain BCE would let the
    flood of normal beats drown them out. IMPORTANT: pred here is a PROBABILITY
    in (0, 1) (the model already applied sigmoid), so BCE is re-implemented by
    hand rather than via BCEWithLogitsLoss, which expects raw logits.

    gamma : focusing strength (0 = plain weighted BCE; typical 2.0).
    alpha : positive-class weight in [0, 1] (0.25 favors the negatives, which
            dominate the multi-hot labels).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # eps inside both logs prevents log(0) = -inf when a probability is
        # exactly 0 or 1; this is the numerical-stability guard for the manual
        # BCE in probability space.
        eps = 1e-8
        bce = -(target * torch.log(pred + eps) +
                (1.0 - target) * torch.log(1.0 - pred + eps))
        # pt = the probability the model put on the TRUE answer for each cell
        # (pred for positives, 1-pred for negatives). (1-pt)^gamma is the focal
        # modulation that shrinks the loss on confident-correct cells.
        pt = torch.where(target == 1, pred, 1.0 - pred)
        # alpha_t broadcasts alpha to positives and (1-alpha) to negatives.
        alpha_t = torch.where(target == 1,
                              torch.full_like(pred, self.alpha),
                              torch.full_like(pred, 1.0 - self.alpha))
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


class LabelSmoothBCELoss(nn.Module):
    """BCE with label smoothing: 1 -> (1-eps), 0 -> eps/2.

    # Intuition: https://arxiv.org/abs/1906.02629

    Softens the hard 0/1 targets so the model is never pushed all the way to
    100% or 0% certainty -- a mild regularizer that tends to give better-
    calibrated probabilities, which matters when the output is meant to be read
    as a clinical likelihood. Operates on probabilities (sigmoid already
    applied), like FocalLoss.

    smoothing : amount of mass moved toward 0.5 (typical 0.1). With smoothing=s,
                a positive target becomes 1 - s/2 and a negative becomes s/2
                (the formula below evaluates to those endpoints).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8                     # log(0) guard, same as FocalLoss
        # Smoothed target: t = target*(1-s) + 0.5*s. For target=1 -> 1 - s/2;
        # for target=0 -> s/2. Symmetric pull of both classes toward 0.5.
        t = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return (-(t * torch.log(pred + eps) +
                  (1.0 - t) * torch.log(1.0 - pred + eps))).mean()


class WeightedBCELoss(nn.Module):
    """Per-class pos_weight BCE operating on probabilities.

    Scales the positive term of each class by a precomputed weight (the
    neg/pos ratio, see build_loss_fn) so a rare diagnosis counts for more --
    a deliberate counterweight to the imbalance where most ECGs are negative
    for any given class. Probability-domain BCE like the others.

    pos_weight : (C,) tensor, one multiplier per class.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        # register_buffer (not a Parameter): pos_weight moves with .to(device)
        # and is saved in state_dict, but is NOT updated by the optimizer.
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8                     # log(0) guard

        pw = self.pos_weight.to(pred.device)

        # Only the positive term is scaled by pw; the negative term keeps weight 1.
        return (-(pw * target * torch.log(pred + eps) +
                  (1.0 - target) * torch.log(1.0 - pred + eps))).mean()


# -- Normalization ----------------------------------------------------------

def normalize_batch(x: torch.Tensor, strategy: str) -> torch.Tensor:
    """
    Per-batch input normalization.

    Parameters
    ----------
    x : torch.Tensor
        (B, seq=59, features=2892) after the reshape from (B, 12, 241, 59).
        The 2892 features are 12 leads x 241 frequency bins, in that nesting
        order (lead-major), which is what the "per_channel" view below relies on.
    strategy : str
        "none"               : pass-through, no normalization.
        "zero_mean_unit_var" : per-ECG z-score over all seq and feature dims
                               (one mean/std per ECG) -- the default that
                               follows the original study.
        "per_channel"        : z-score each of the 12 leads independently.

    Returns
    -------
    torch.Tensor of the same shape (B, 59, 2892).

    Every branch adds 1e-8 to the std before dividing -- the numerical-stability
    guard against a zero-variance (constant) lead/ECG blowing up to inf, as can
    happen on a flatline lead.
    """
    if strategy == "none":
        return x

    elif strategy == "zero_mean_unit_var":
        # Per-ECG global mean/std across all seq and feature dims.
        # keepdim=True keeps the (B, 1, 1) shape so it broadcasts back over x.
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2),  keepdim=True) + 1e-8   # +eps: avoid /0 on a constant ECG
        return (x - mean) / std

    elif strategy == "per_channel":
        # Normalize each of the 12 ECG leads independently.
        # Reinterpret the flat 2892 feature axis as (12 leads, 241 freq) WITHOUT
        # moving data (view is free) -- this only works because the reshape in
        # train_epoch laid the features out lead-major (lead, then freq).
        B, S, C = x.shape  # C = 12 * 241 = 2892
        x = x.view(B, S, 12, 241)
        # Reduce over seq (1) and freq (3), leaving a stat per (batch, lead).
        mean = x.mean(dim=(1, 3), keepdim=True)  # (B, 1, 12, 1)
        std  = x.std(dim=(1, 3),  keepdim=True)  + 1e-8
        # Normalize, then collapse back to the original flat feature axis.
        return ((x - mean) / std).view(B, S, C)

    else:
        raise ValueError(f"Unknown normalization strategy: {strategy!r}")


# -- Builder helpers --------------------------------------------------------

def build_loss_fn(cfg, train_loader, device) -> nn.Module:
    """
    Construct the loss module selected by cfg.training.loss_fn.

    All four losses operate on SIGMOID PROBABILITIES, matching the model output
    (the model applies sigmoid in its forward); nn.BCELoss likewise expects
    probabilities, not logits.

    Returns an nn.Module; raises ValueError on an unknown name.
    """
    name = getattr(cfg.training, "loss_fn", "bce")   # default = plain BCE (the choice of the original study)

    if name == "bce":
        return nn.BCELoss()

    if name == "focal":
        gamma = float(getattr(cfg.training, "focal_gamma", 2.0))
        alpha = float(getattr(cfg.training, "focal_alpha", 0.25))
        return FocalLoss(gamma=gamma, alpha=alpha)

    if name == "bce_smooth":
        smoothing = float(getattr(cfg.training, "label_smoothing", 0.1))
        return LabelSmoothBCELoss(smoothing=smoothing)

    if name == "bce_weighted":
        # One full sweep over the train set to count positives per class.

        pos   = torch.zeros(cfg.model.num_classes)   # running positive count per class
        total = 0                                    # running ECG count
        for _, labels in train_loader:
            pos   += labels.float().sum(0)           # sum positives down the batch axis
            total += labels.size(0)
        neg        = total - pos                     # negatives = total - positives, per class

        # pos_weight = neg/pos (the classic imbalance ratio). +1e-8 avoids /0 for
        # a class with zero positives; clamp(0.1, 10.0) caps the weight so a very
        # rare class can't dominate the gradient and destabilize training.
        pos_weight = (neg / (pos + 1e-8)).clamp(0.1, 10.0)

        return WeightedBCELoss(pos_weight.to(device))

    raise ValueError(f"Unknown loss_fn: {name!r}")


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    """
    Build the optimizer selected by cfg.training.optimizer ("adam" | "adamw").

    Adam applies decay coupled into the gradient, AdamW decouples it (true weight decay).
    """
    name = getattr(cfg.training, "optimizer", "adam")
    lr   = float(cfg.training.learning_rate)
    wd   = float(getattr(cfg.training, "weight_decay", 0.0))   # default 0 = no decay

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(cfg, optimizer, n_batches_per_epoch: int):
    """
    Returns (scheduler, mode).
    mode = "epoch"   -> call scheduler.step() once per epoch
    mode = "plateau" -> call scheduler.step(val_auroc) once per epoch

    NOTE on the annealing schedules ("cosine", "cosine_warmup", "onecycle"):
    T_max / total_steps are set to cfg.training.num_epochs, which assumes the
    run executes all num_epochs. With early stopping enabled
    (early_stopping_patience << num_epochs, e.g. patience=5 / num_epochs=500),
    training stops after ~20-30 epochs, so the cosine curve never reaches its
    trough and the schedule behaves as a warmup-to-near-constant-LR.

    Parameters
    ----------
    cfg : DictConfig
        Reads cfg.training.lr_scheduler and the per-schedule knobs below.
    optimizer : torch.optim.Optimizer
        The optimizer whose LR this schedule drives.
    n_batches_per_epoch : int
        Batches per epoch.

    Returns
    -------
    (scheduler, mode) where mode in {"epoch", "plateau"}; raises ValueError on
    an unknown scheduler name. The caller dispatches on mode.
    """
    # NOTE: n_batches_per_epoch is accepted but not used -- every schedule below
    #       is stepped once per epoch.
    name       = getattr(cfg.training, "lr_scheduler", "reduce_lr")
    num_epochs = int(cfg.training.num_epochs)

    if name == "reduce_lr":
        # mode="max": LR drops when the watched metric (val AUROC) stops rising.
        # This is the only schedule that returns "plateau" -- the loop feeds it
        # val_auroc rather than calling a bare step().
        return (ReduceLROnPlateau(optimizer, mode="max",
                                  factor=float(cfg.training.lr_decay_factor),
                                  patience=int(cfg.training.lr_scheduler_patience)),
                "plateau")

    if name == "cosine":
        # Smooth cosine decay from the base LR down to eta_min over num_epochs.
        return (CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6),
                "epoch")

    if name == "cosine_warmup":
        # Linear warmup (0.1x -> 1x base LR over warmup epochs) then cosine
        # decay for the rest. SequentialLR chains the two at milestone=warmup.
        warmup = int(getattr(cfg.training, "warmup_epochs", 10))
        return (SequentialLR(optimizer,
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.1,
                                          end_factor=1.0, total_iters=warmup),
                                 # max(1, ...) guards against warmup >= num_epochs
                                 # making T_max zero (which CosineAnnealingLR rejects).
                                 CosineAnnealingLR(optimizer,
                                                   T_max=max(1, num_epochs - warmup),
                                                   eta_min=1e-6),
                             ],
                             milestones=[warmup]),
                "epoch")

    if name == "onecycle":
        max_lr = float(getattr(cfg.training, "lr_max_onecycle", 0.001))
        # pct_start=0.1: 10% of the cycle ramps up, 90% anneals down.
        return (OneCycleLR(optimizer, max_lr=max_lr,
                           total_steps=num_epochs, pct_start=0.1),
                "epoch")

    if name == "step":
        # Multiply LR by gamma every step_size epochs. This is the main
        # recipe's schedule and is immune to the early-stop caveat.
        step_size = int(getattr(cfg.training, "step_lr_step_size", 50))
        gamma     = float(getattr(cfg.training, "step_lr_gamma", 0.5))
        return (StepLR(optimizer, step_size=step_size, gamma=gamma),
                "epoch")

    raise ValueError(f"Unknown lr_scheduler: {name!r}")