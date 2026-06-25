"""
scripts/sweep_mask_ablation.py
==================
The final replication sweep. It runs every experiment the original study
(Kang et al., 2025) reports that I had not yet covered, plus a handful of
educated-guess variations aimed at closing the gap I still have against the
study's headline number.

Each phase maps to a section of the original study or to one specific
gap-closing test. The study's recipe is the fixed anchor everything is
measured against: Adam, StepLR(step=2, gamma=0.8), lr=2e-4, dropout=0.5,
mask 0.2/0.8, embedding_dim=256, num_blocks=2, fusion=layer, patience=5.

1 'ptbxl_paper_exact'      - the Section 4.3 main result, one run x 3 seeds.
                             I had never actually run this: my earlier PTB-XL
                             'best' used a different architecture entirely
                             (slstm_only, nb=1, AdamW, cosine_warmup).
2 'table5_fixed_depth'      - Table 5 done cleanly, all 4 fusion variants at the
                             *same* depth (nb=2). My earlier fusion comparison
                             mixed nb=1 and nb=2 across variants, so its
                             'sLSTM-only beats layer fusion' result was
                             confounded by parameter count rather than fusion.
4 'table3_mask_ratio'       - the Table 3 search: mask_ratio in
                             {0.1, 0.2, 0.3, 0.4, 0.5} x 3 seeds at the study
                             recipe with patience=2 (its Section 4.4
                             hyper-search protocol).
5 'table4_mask_prob'        - Table 4: mask_prob in {0.6, 0.7, 0.8, 0.9, 1.0}
                             x 3 seeds, same hyper-search protocol.
6 'patience_15'             - the study recipe but with patience=15. Tests
                             whether its patience=5 stops training too soon.
7 'cosine_warmup_variant'   - the study recipe but with a cosine-warmup LR
                             schedule. An educated guess: cosine warmup beat
                             StepLR in my earlier (non-study) searches, so if it
                             also helps under the faithful recipe I may close
                             part of the gap without departing from the study.
8 'ensemble_extra_seeds'    - 7 more seeds of the exact Phase 1 recipe, so I can
                             build a 10-seed ensemble for the final report.
9 'georgia_paper_recipe'    - the study-exact recipe transferred to Georgia under
                             its strict split (g2..g11 train, g1 test).
                             nb in {2, 4, 6} x 3 seeds = 9 runs. A strict
                             replication of Section 4.7 + Table 7 of the study.


Usage
-----
    python scripts/sweep_mask_ablation.py --gpus 0,0,0,1,1,1 --dry-run
    python scripts/sweep_mask_ablation.py --gpus 0,0,0,1,1,1 2>&1 | tee logs/sweep_mask_ablation.log
    python scripts/sweep_mask_ablation.py --gpus 0,0,0,1,1,1 --phases 1,2  # subset


"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT   = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT / "results" / "sweep_mask_ablation"

# Reuse the *same* interpreter that launched the harness for the child runs, so
# they share my virtualenv (and its exact torch/CUDA build) instead of some
# system python that might resolve to a different environment.
PYTHON    = sys.executable


# The study's PTB-XL main config (Section 4.3 + Section 3.1 of the original
# study), written as OmegaConf dotlist overrides ("section.key": value).
PAPER_PTBXL = {
    "training.optimizer":               "adam",
    "training.learning_rate":           2e-4,
    "training.lr_scheduler":            "step",
    "training.step_lr_step_size":       2,
    "training.step_lr_gamma":           0.8,
    "training.weight_decay":            0.0,
    "training.normalization":           "zero_mean_unit_var",
    "training.loss_fn":                 "bce",
    "training.early_stopping_patience": 5,
    # num_epochs is deliberately huge: the original study trains to early
    # stopping, so I never want the epoch cap to be what ends a run. patience=5
    # (above) is the real stopping criterion.
    "training.num_epochs":              500,        # rely on early stop
    "model.embedding_dim":              256,
    "model.num_heads":                  4,
    "model.num_blocks":                 2,          # two fused blocks (Section 4.3 of the original study)
    "model.fusion_type":                "layer",    # eq. 13 of the original study: mutual-refinement fusion
    "model.dropout":                    0.5,
    "model.mask_ratio":                 0.2,        # Table 3 default (fraction of seq positions eligible to mask)
    "model.mask_prob":                  0.8,        # Table 4 default (per-sample prob masking is applied to a sample)
    "model.pooling":                    "mean",
}

# Hyperparameter-search protocol (Section 4.4 of the original study): identical
# to PAPER_PTBXL except patience=2 instead of 5. The study's Table 2/3/4 grid
# searches use the shorter patience so each grid point trains quickly; the main
# result (PAPER_PTBXL) uses the longer patience=5. The dict-unpack + override is
# order-sensitive: the explicit patience key below MUST come after **PAPER_PTBXL
# so it wins.
HYPER_SEARCH_PROTOCOL = {
    **PAPER_PTBXL,
    "training.early_stopping_patience": 2,
}

# Georgia transfer overrides (Section 4.7 / Table 7 of the original study).
# These switch the dataset branch in train.py (dataset_type="georgia"), point at
# the on-disk Georgia WFDB and its STFT cache, select the study's strict group
# split, and fix the head to the 7 SNOMED-CT rhythm classes. They are layered ON
# TOP of PAPER_PTBXL in phase9, so the keys here deliberately
# override the PTB-XL training schedule for the Georgia runs.
GEORGIA_DATA_PAPER_STRICT = {
    "data.dataset_type":           "georgia",
    # Georgia data paths come from config.yaml (set data.root there); I leave them
    # un-overridden here on purpose, so there is one single place to point at data.
    "data.georgia_split_strategy": "paper_strict",  # test=g1, train=g2..g11 (val re-uses the train pool; no held-out val)
    "model.num_classes":           7,               # NSR/AF/IAVB/LBBB/RBBB/SB/STach
    "training.num_epochs":         20,              # the original study fixes 20 epochs, no early stop
    # patience=999 effectively disables early stopping so the run always trains
    # the full 20 epochs (Section 4.7 of the original study reports a fixed-epoch
    # schedule).
    "training.early_stopping_patience": 999,        # disabled - fixed 20 epochs per the original study
}

# Three base seeds give a mean +/- std for every reported cell. The model has
# genuine cross-seed variance (the bfloat16 sLSTM kernel is non-deterministic
# across processes), so any single-seed number would be misleading.
SEEDS_THREE = [42, 123, 456]
# Seven *more* seeds run with the exact Phase 1 recipe; together with the three
# above they give the 10 checkpoints ensemble_eval.py averages.
SEEDS_EXTRA = [789, 999, 11, 13, 17, 23, 29]   # for 10-seed ensemble (3 + 7)


# ── Phase configs ────────────────────────────────────────────────────────────

def phase1_configs() -> list[dict]:
    """Build the Phase 1 run specs: the study-exact PTB-XL recipe over 3 seeds.

    Each returned dict is a *run spec* with three keys the worker consumes:

    - 'name'      - unique run id; also the per-run directory name under
                    results/sweep_mask_ablation/.
    - 'group'     - the coarse bucket _print_group_summary uses to compute a
                    mean +/- std across the runs that share it.
    - 'overrides' - the OmegaConf delta handed to run_training.

    Returns
    -------
    list[dict]
        Exactly len(SEEDS_THREE) (== 3) specs. All share the same group, so they
        aggregate into one reported cell.
    """
    return [
        # Only training.seed varies across the three; everything else is the
        # frozen study recipe. Same group => one mean +/- std cell in the summary.
        {"name": f"ptbxl_paper_exact_s{s}", "group": "ptbxl_paper_exact",
         "overrides": {**PAPER_PTBXL, "training.seed": s}}
        for s in SEEDS_THREE
    ]


def phase2_configs() -> list[dict]:
    """Build the fusion ablation at a FIXED depth nb=2.

    Returns
    -------
    list[dict]
        4 fusion variants x len(SEEDS_THREE) = 12 specs. Each fusion is its own
        group, so the summary reports one mean +/- std per variant.
    """
    out = []
    for fusion in ("slstm_only", "mlstm_only", "sequential", "layer"):
        for s in SEEDS_THREE:
            out.append({
                "name": f"table5_{fusion}_nb2_s{s}",
                "group": f"table5_{fusion}",
                # num_blocks=2 is pinned here even though PAPER_PTBXL already
                # sets it - the explicit pin documents that depth is held fixed.
                # NOTE: for fusion="sequential", nb>=2 is also what makes it
                # differ from "layer" at all (at nb=1 the two are identical).
                "overrides": {**PAPER_PTBXL,
                              "model.fusion_type": fusion,
                              "model.num_blocks":  2,
                              "training.seed":     s},
            })
    return out


def phase4_configs() -> list[dict]:
    """Build the Phase 4 (Table 3) mask-RATIO search.

    mask_ratio controls *how much* of each sequence is eligible for the random
    masking augmentation - the fraction of the 59 time positions placed in the
    candidate pool. This grid uses HYPER_SEARCH_PROTOCOL (patience=2), matching
    the study's Section 4.4 hyper-search budget.

    Returns
    -------
    list[dict]
        5 ratios x len(SEEDS_THREE) = 15 specs, one group per ratio.
    """
    out = []
    # NOTE: ratio=0.2 reproduces the study's chosen default, so it doubles as a
    # consistency check against Phase 1 (modulo patience=2 here vs 5 there).
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        for s in SEEDS_THREE:
            out.append({
                "name": f"table3_mask_ratio_{r}_s{s}",
                "group": f"table3_mask_ratio_{r}",
                "overrides": {**HYPER_SEARCH_PROTOCOL,
                              "model.mask_ratio": r,
                              "training.seed":    s},
            })
    return out


def phase5_configs() -> list[dict]:
    """Build the Phase 5 (Table 4) mask-PROBABILITY search.

    mask_prob is the *per-position* probability that an eligible position is
    actually masked once it is in the candidate pool - distinct from mask_ratio,
    which sets the size of that pool. At mask_prob=1.0 every eligible position is
    masked. Same patience=2 hyper-search protocol as Phase 4.

    Returns
    -------
    list[dict]
        5 probabilities x len(SEEDS_THREE) = 15 specs, one group each.
    """
    out = []
    # NOTE: prob=0.8 reproduces the study default (cross-checks Phase 1);
    # prob=1.0 is the "always mask the eligible positions" extreme.
    for p in (0.6, 0.7, 0.8, 0.9, 1.0):
        for s in SEEDS_THREE:
            out.append({
                "name": f"table4_mask_prob_{p}_s{s}",
                "group": f"table4_mask_prob_{p}",
                "overrides": {**HYPER_SEARCH_PROTOCOL,
                              "model.mask_prob": p,
                              "training.seed":   s},
            })
    return out


def phase6_configs() -> list[dict]:
    """Build Phase 6: the study recipe but with early-stopping patience=15.

    A controlled probe of whether the study's patience=5 stops too early. Only
    the patience knob differs from PAPER_PTBXL; the explicit key after
    **PAPER_PTBXL overrides the inherited patience=5.

    Returns
    -------
    list[dict]
        len(SEEDS_THREE) = 3 specs sharing one group.
    """
    return [
        {"name": f"ptbxl_paper_patience15_s{s}",
         "group": "ptbxl_paper_patience15",
         "overrides": {**PAPER_PTBXL,
                       "training.early_stopping_patience": 15,
                       "training.seed": s}}
        for s in SEEDS_THREE
    ]


def phase7_configs() -> list[dict]:
    """Build Phase 7: the study recipe but with a cosine-warmup LR schedule.

    An educated-guess deviation from the study's StepLR: cosine warmup beat
    StepLR in my earlier (non-study) searches, so I test whether it still helps
    under the otherwise-faithful recipe. warmup_epochs=10 is only read when
    lr_scheduler == "cosine_warmup" (inert otherwise), so it is harmless to set,
    but I set it explicitly for reproducibility.

    Returns
    -------
    list[dict]
        len(SEEDS_THREE) = 3 specs sharing one group.
    """
    return [
        {"name": f"ptbxl_paper_cosine_s{s}",
         "group": "ptbxl_paper_cosine",
         "overrides": {**PAPER_PTBXL,
                       "training.lr_scheduler": "cosine_warmup",
                       "training.warmup_epochs": 10,
                       "training.seed": s}}
        for s in SEEDS_THREE
    ]


def phase8_configs() -> list[dict]:
    """Build Phase 8: 7 extra seeds of the exact Phase 1 recipe.

    Deliberately reuses Phase 1's name/group scheme (ptbxl_paper_exact) with the
    SEEDS_EXTRA seeds. Together with Phase 1's three runs this gives ten
    checkpoints under the same group, which ensemble_eval.py later globs
    (ptbxl_paper_exact_s*/checkpoints/best.pt) to build the 10-seed ensemble.

    Because the names follow the same ptbxl_paper_exact_s{seed} pattern, a
    Phase 1 run that already finished would be detected as "done" and skipped if
    its seed overlapped - but SEEDS_THREE and SEEDS_EXTRA are disjoint, so in
    practice all 7 here are fresh.

    Returns
    -------
    list[dict]
        len(SEEDS_EXTRA) = 7 specs sharing the Phase 1 group.
    """
    return [
        {"name": f"ptbxl_paper_exact_s{s}", "group": "ptbxl_paper_exact",
         "overrides": {**PAPER_PTBXL, "training.seed": s}}
        for s in SEEDS_EXTRA
    ]


def phase9_configs() -> list[dict]:
    """Build Phase 9: the study-exact recipe transferred to Georgia.

    A strict replication of the Section 4.7 / Table 7 transfer in the original
    study: the same optimizer/LR/model recipe as PTB-XL, but on the Georgia
    dataset under the strict group split, sweeping depth num_blocks in {2, 4, 6}.

    Override layering is load-bearing here. PAPER_PTBXL is unpacked first, then
    GEORGIA_DATA_PAPER_STRICT second, so on the keys they share
    (training.num_epochs, training.early_stopping_patience) the Georgia values
    win - i.e. the run trains for a fixed 20 epochs with early stopping disabled,
    not the PTB-XL 500-epoch / patience-5 schedule. model.num_blocks is then
    pinned last so it overrides the depth carried by either base dict.

    Returns
    -------
    list[dict]
        3 depths x len(SEEDS_THREE) = 9 specs, all sharing one group (depth lives
        only in name, so the group mean spans all depths; per-depth reads come
        from the names).
    """
    out = []
    # The Georgia overrides come AFTER PAPER_PTBXL on purpose, so they win on the
    # shared keys (num_epochs, early_stopping_patience).
    for nb in (2, 4, 6):
        for s in SEEDS_THREE:
            out.append({
                "name": f"georgia_paper_recipe_nb{nb}_s{s}",
                "group": "georgia_paper_recipe",
                # num_blocks pinned LAST so it beats the depth in both base
                # dicts; a dict literal evaluates left-to-right, last write wins.
                "overrides": {**PAPER_PTBXL, **GEORGIA_DATA_PAPER_STRICT,
                              "model.num_blocks": nb,
                              "training.seed":    s},
            })
    return out



def worker(gpu_id: int, work_queue: queue.Queue,
           results: list, lock: threading.Lock, start_time: float):
    """Thread body: pull run specs off the shared queue and run them on one GPU.

    One worker is spawned per entry in the --gpus list, so '0,0,0,1,1,1' gives
    three workers bound to GPU 0 and three to GPU 1. All workers drain a single
    shared queue, which load-balances naturally: a worker that finishes a short
    run grabs the next spec immediately instead of idling behind a slower sibling.

    Parameters
    ----------
    gpu_id : int
        Physical GPU index; passed to each child via CUDA_VISIBLE_DEVICES.
    work_queue : queue.Queue
        Shared queue of run-spec dicts (the output of the phase builders).
    results : list
        Shared accumulator the runs append their result dicts to (guarded by
        lock).
    lock : threading.Lock
        Serializes appends to results and the summary rewrite.
    start_time : float
        time.time() at sweep start, used only for the "+H.HHh" progress stamps.
    """
    while True:
        try:
            # get_nowait (not get()) so a worker exits the instant the queue is
            # empty instead of blocking forever on work that never comes - there
            # is no sentinel / poison pill in this design.
            spec = work_queue.get_nowait()
        except queue.Empty:
            break
        _run_training(gpu_id, spec, results, lock, start_time)
        # task_done balances the implicit get; harmless here since I join the
        # threads (not the queue), but kept so queue.join() would also work.
        work_queue.task_done()


def _run_training(gpu_id: int, spec: dict,
                  results: list, lock: threading.Lock, start_time: float):
    """Run (or resume) a single training job as a run_training subprocess.

    Parameters
    ----------
    gpu_id : int
        GPU index to pin this child to (via CUDA_VISIBLE_DEVICES).
    spec : dict
        Run spec with name, group, overrides (see the phase builders).
    results, lock, start_time
        Shared accumulator / its lock / sweep-start timestamp (see worker).

    """
    name      = spec["name"]
    group     = spec["group"]
    overrides = spec["overrides"]

    # One directory per run; its presence is also our resume sentinel.
    run_dir     = SWEEP_DIR / name
    result_file = run_dir / "result.json"   # written by the child train.py
    log_file    = run_dir / "train.log"     # wrapper log written by THIS process
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume path: a prior run already produced metrics -> do not retrain.
    if result_file.exists():
        elapsed_h = (time.time() - start_time) / 3600
        print(f"[GPU{gpu_id}] SKIP  {name:<48} +{elapsed_h:.2f}h (done)", flush=True)
        with open(result_file) as f:
            result = json.load(f)

        # Re-stamp the harness-side bookkeeping in case the child's result.json
        # didn't carry it (the child only knows its config, not my run name).
        result.update({"name": name, "group": group, "overrides": overrides})
        with lock:
            results.append(result)
        # NOTE: no summary.json rewrite on the skip path - it gets rebuilt from
        # results at the end of each phase anyway, and skips are cheap.
        return

    # ---- Fresh run: tell the child WHERE to write its three outputs. These are
    # the only config keys the harness injects (everything else is recipe).
    # training.result_file makes train.py emit a metrics
    # JSON - without it the child writes no result.json.
    extra = [
        f"training.checkpoint_dir={run_dir / 'checkpoints'}",
        f"training.log_dir={run_dir / 'logs'}",
        f"training.result_file={result_file}",
    ]
    # Build the argv: python -m scripts.run_training k1=v1 k2=v2 ... <extra>.
    # The overrides come first so a stray recipe key of the same name can't
    # clobber the harness-injected output paths.
    cmd = ([PYTHON, "-m", "scripts.run_training"]
           + [f"{k}={v}" for k, v in overrides.items()] + extra)
    # Pin the child to exactly one GPU. Copy the full parent env (PATH, CUDA
    # libs, virtualenv) and override only the device mask, so several children
    # on different GPUs don't contend for the same card.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    elapsed_h = (time.time() - start_time) / 3600
    print(f"[GPU{gpu_id}] START {name:<48} +{elapsed_h:.2f}h", flush=True)

    t0 = time.time()
    try:
        with open(log_file, "w") as log_f:
            # Header so the log is self-describing: exact command + GPU + start.
            log_f.write(f"CMD: {' '.join(str(c) for c in cmd)}\n")
            log_f.write(f"GPU: {gpu_id}\nSTARTED: {datetime.now().isoformat()}\n\n")
            log_f.flush()  # flush header before the child starts appending
            # Merge the child's stdout AND stderr into this one log file so a
            # crash traceback lands next to the training output that preceded it.
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f,
                                     cwd=str(PROJECT))
            try:
                # 10800 s = 3 h hard cap per run. Generous: even the deepest
                # Georgia nb=6 run finishes well inside this. The cap is only
                # there to stop a hung kernel from wedging a GPU worker forever.
                proc.wait(timeout=10800)
            except subprocess.TimeoutExpired:
                # Kill the child, then wait() again to reap the zombie before
                # re-raising into the outer handler. Magick.
                proc.kill(); proc.wait()
                raise

        elapsed = time.time() - t0
        # Success means the child wrote result.json, NOT a 0 return code alone -
        # a clean exit with no metrics still counts as a FAIL.
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            result.update({"name": name, "group": group, "overrides": overrides})
            # test_auroc is the one field every downstream summary reads; default
            # to NaN so a malformed result.json prints rather than raising KeyError.
            auroc = result.get("test_auroc", float("nan"))
            print(f"[GPU{gpu_id}] DONE  {name:<48} "
                  f"AUROC={auroc:.4f}  {elapsed/60:.1f}min", flush=True)
        else:
            # Child exited (possibly 0) but produced no metrics -> record the rc
            # so the failure can be diagnosed from summary.json without the log.
            result = {"name": name, "group": group, "overrides": overrides,
                      "error": "no result.json", "rc": proc.returncode}
            print(f"[GPU{gpu_id}] FAIL  {name:<48} rc={proc.returncode}", flush=True)

    except subprocess.TimeoutExpired:
        # Re-raised from the inner kill-and-wait above.
        result = {"name": name, "group": group, "overrides": overrides,
                  "error": "timeout"}
        print(f"[GPU{gpu_id}] TIMEOUT {name}", flush=True)
    except Exception as e:
        # Catch-all so an unexpected harness-side error (bad path, OOM at spawn,
        # JSON decode error on a truncated result.json) still records a row
        # instead of killing the worker thread.
        result = {"name": name, "group": group, "overrides": overrides,
                  "error": str(e)}
        print(f"[GPU{gpu_id}] ERROR {name}: {e}", flush=True)

    # Append the row and rewrite the rolling summary so the
    # leaderboard on disk stays consistent even mid-sweep.
    with lock:
        results.append(result)
        _write_summary(results)


def _is_done(spec: dict) -> bool:
    """Return True iff this run already produced a result.json on disk.

    Used by run_phase to size the pending work and to print "N already done";
    mirrors the same sentinel check _run_training makes. A pure filesystem
    probe - it never reads the file contents.
    """

    # Constructs the path <SWEEP_DIR>/<spec name>/result.json.
    return (SWEEP_DIR / spec["name"] / "result.json").exists()


def _write_summary(results: list):
    """Rewrite results/sweep_mask_ablation/summary.json as a leaderboard snapshot.

    Called after every completed run (inside _run_training's lock) so the
    on-disk summary is always current and the sweep can be watched live. The
    whole file is rewritten each time.

    Partition rule: a result counts as a finished training run iff it has a
    'test_auroc' key; anything else with an 'error' key is a failure.

    Parameters
    ----------
    results : list
        The shared accumulator of per-run result dicts.
    """
    # Completed runs, sorted best-AUROC-first so the top of the file is the
    # current leader. The key default of 0 is unreachable here (already filtered
    # on "test_auroc" in r) but keeps the lambda total.
    done = sorted([r for r in results if "test_auroc" in r],
                  key=lambda r: r.get("test_auroc", 0), reverse=True)
    fail = [r for r in results if "test_auroc" not in r and "error" in r]
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump({
            "updated":  datetime.now().isoformat(),
            "training": {"completed": len(done), "failed": len(fail), "runs": done},
            "all_failed": fail,
        }, f, indent=2)


def _print_group_summary(results: list, group_substr: str):
    """Print a per-group mean +/- std AUROC table for one phase to stdout.

    Parameters
    ----------
    results : list
        Shared accumulator; only entries with test_auroc are considered.
    group_substr : str
        Prefix matched against each run's group field.

    """
    runs = [r for r in results if r.get("group", "").startswith(group_substr)
            and "test_auroc" in r]
    if not runs:
        return
    aurocs = [r["test_auroc"] for r in runs]
    mean = sum(aurocs) / len(aurocs)
    # Population std (divide by N, not N-1): a descriptive spread over the seeds
    # I actually ran, not an inferential estimate of a wider population.
    # NOTE: the reported 95% CIs come from bootstrap_ci.py, not from this std.
    std  = (sum((x - mean) ** 2 for x in aurocs) / len(aurocs)) ** 0.5
    print(f"\n  group=~{group_substr} ({len(runs)} runs):")
    for r in sorted(runs, key=lambda x: x["name"]):
        print(f"    {r['name']:<50}  AUROC={r['test_auroc']:.4f}")
    print(f"    Mean = {mean:.4f}  Std = {std:.4f}")


def run_phase(phase_name: str, jobs: list[dict], gpu_ids: list[int],
              results: list, lock: threading.Lock,
              start_time: float, dry_run: bool):
    """Run all jobs of one phase across the GPU worker pool (or dry-run them).

    Parameters
    ----------
    phase_name : str
        Human-readable phase label for the log header.
    jobs : list[dict]
        The phase's run specs (output of a phaseN_configs builder).
    gpu_ids : list[int]
        Parsed --gpus list; its length is the worker count.
    results, lock, start_time
        Shared accumulator / lock / sweep-start timestamp, threaded to workers.
    dry_run : bool
        If True, print the planned diffs and return without launching anything.
    """
    if not jobs:
        print(f"\n[sweep_mask_ablation] Phase '{phase_name}': no jobs.")
        return
    # Count how many are already on disk so the header reflects resume progress.
    pending = [j for j in jobs if not _is_done(j)]
    print(f"\n[sweep_mask_ablation] Phase '{phase_name}': "
          f"{len(pending)}/{len(jobs)} jobs to run "
          f"({len(jobs) - len(pending)} already done)", flush=True)

    if dry_run:
        for j in jobs:
            # Show only the keys whose value departs from PAPER_PTBXL: a key the
            # study dict doesn't have at all (e.g. data.* / warmup_epochs), OR a
            # key whose value I changed (e.g. mask_ratio, seed, patience). This
            # is the "diff from the study recipe" view, not the full override.
            mut = {k: v for k, v in j["overrides"].items()
                   if k not in PAPER_PTBXL or PAPER_PTBXL.get(k) != v}
            print(f"  TRAIN {j['name']:<50}  diff_from_paper={mut}")
        return

    # Enqueue ALL jobs (including already-done ones); the resume check inside
    # _run_training turns finished jobs into fast SKIPs, so re-queuing them is
    # harmless and saves this function from special-casing resumes.
    wq = queue.Queue()

    for j in jobs: wq.put(j)
    # One worker thread per --gpus entry. daemon=True so a Ctrl-C on the main
    # thread doesn't hang on stuck workers; the name encodes GPU + slot for logs.
    threads = [threading.Thread(target=worker,
                                 args=(g, wq, results, lock, start_time),
                                 daemon=True, name=f"GPU{g}-w{i}")
               for i, g in enumerate(gpu_ids)]
    for t in threads: t.start()

    # Block until every worker has drained the queue and exited -> the phase is
    # fully complete before main() prints its group summary and moves on.
    for t in threads: t.join()


# Phase registry: maps the CLI phase number to a
# (log_label, config_builder, summary_group_prefix) triple. main iterates the
# requested phase numbers, calls the builder to get the jobs, runs them, then
# prints the group summary keyed by the prefix.
#
# Two intentional gaps in the numbering, both because those steps are separate
# scripts that run AFTER this sweep (not phases of it):
#   - Phase 3  = per-class threshold tuning  (scripts/threshold_tune.py)
#   - Phase 10 = final ensemble eval / CIs   (ensemble_eval.py, bootstrap_ci.py)
#
# Note Phase 8 deliberately reuses Phase 1's "ptbxl_paper_exact" prefix: its
# extra seeds aggregate into the SAME reported group as Phase 1 (that is the
# whole point - together they form the 10-seed set).
PHASES = {
    1: ("1_ptbxl_paper_exact",        phase1_configs, "ptbxl_paper_exact"),
    2: ("2_table5_fixed_depth",       phase2_configs, "table5_"),
    4: ("4_table3_mask_ratio",        phase4_configs, "table3_mask_ratio"),
    5: ("5_table4_mask_prob",         phase5_configs, "table4_mask_prob"),
    6: ("6_ptbxl_paper_patience15",   phase6_configs, "ptbxl_paper_patience15"),
    7: ("7_ptbxl_paper_cosine",       phase7_configs, "ptbxl_paper_cosine"),
    8: ("8_ensemble_extra_seeds",     phase8_configs, "ptbxl_paper_exact"),
    9: ("9_georgia_paper_recipe",     phase9_configs, "georgia_paper_recipe"),
}


def main():
    p = argparse.ArgumentParser(description="sweep_mask_ablation: final paper-replication")
    p.add_argument("--gpus",    default="0,0,0,1,1,1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--phases",  default="1,2,4,5,6,7,8,9",
                   help="Comma-separated phase numbers (default all). "
                        "Phase 3 (threshold tuning) and Phase 10 (final eval) "
                        "run as separate scripts after this sweep finishes.")
    args = p.parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    # sorted() so phases always run in ascending order regardless of the order
    # they were typed (e.g. "--phases 9,1" still runs 1 then 9).
    phases  = sorted(int(x) for x in args.phases.split(","))
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"sweep_mask_ablation (final paper replication)  |  phases={phases}  "
          f"|  GPUs: {args.gpus}")
    print(f"Results: {SWEEP_DIR}")
    # Counter collapses the --gpus list into a per-GPU worker tally for the
    # banner (e.g. {0: 3, 1: 3} from 0,0,0,1,1,1).
    counts = Counter(gpu_ids)
    for gid, n in sorted(counts.items()):
        print(f"  GPU{gid}: {n} workers")
    print('=' * 72)

    # Shared state for the whole sweep: one accumulator, one lock guarding it
    # and the summary rewrite, one start timestamp for all the "+H.HHh" stamps.
    results: list = []
    lock = threading.Lock()
    start_time = time.time()

    for ph in phases:
        if ph not in PHASES:
            # Tolerate unknown numbers (e.g. someone passes 3 or 10) instead of
            # erroring - those are the separate-script steps, not sweep phases.
            print(f"[sweep_mask_ablation] Unknown phase {ph}, skipping.")
            continue
        phase_name, builder, group_substr = PHASES[ph]

        run_phase(phase_name, builder(),
                  gpu_ids, results, lock, start_time, args.dry_run)
        if not args.dry_run:
            _print_group_summary(results, group_substr)

    if not args.dry_run:

        elapsed_h = (time.time() - start_time) / 3600
        n_train  = sum(1 for r in results if "test_auroc" in r)
        n_fail   = sum(1 for r in results if "test_auroc" not in r and "error" in r)
        print(f"\nDONE  {elapsed_h:.2f}h  |  {n_train} training runs  "
              f"|  {n_fail} failures")
        # The three eval-only steps are NOT run here - they need all seeds on
        # disk first. I print the exact commands so the operator can copy them.
        # The ensemble glob matches Phase 1 + Phase 8 checkpoints (the 10 seeds).
        print("\nNext steps (run separately):")
        print("  - threshold tuning: python -m scripts.threshold_tune --checkpoint ... --out ...")
        print("  - 10-seed ensemble: python -m scripts.ensemble_eval --checkpoints "
              "results/sweep_mask_ablation/ptbxl_paper_exact_s*/checkpoints/best.pt --out ...")
        print("  - bootstrap CIs   : python -m scripts.bootstrap_ci --checkpoint ... --out ...")
        # NOTE: thread an optional --run-eval flag that chains these three steps
        # automatically once all phases finish, instead of only printing them -
        # right now have to paste the commands by hand.


if __name__ == "__main__":
    main()
