"""
scripts/sweep_main.py
==================
A clean redo of all PTB-XL and Georgia experiments under the corrected
diagnosis-label rule (lik_eq_100).

Usage
-----
    python scripts/sweep_main.py --gpus 0,0,0,1,1,1 --dry-run
    python scripts/sweep_main.py --gpus 0,0,0,1,1,1 2>&1 | tee logs/sweep_main.log
    python scripts/sweep_main.py --gpus 0,0,0,1,1,1 --phases A,B   # subset

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
SWEEP_DIR = PROJECT / "results" / "sweep_main"
PYTHON    = sys.executable


PAPER_PTBXL = {
    "training.optimizer":               "adam",
    "training.learning_rate":           2e-4,    # base LR from the study; StepLR decays it
    "training.lr_scheduler":            "step",  # StepLR (gamma every step_size epochs)
    "training.step_lr_step_size":       2,       # decay every 2 epochs ...
    "training.step_lr_gamma":           0.8,     # ... by x0.8
    "training.weight_decay":            0.0,     # plain Adam, no decoupled decay
    "training.normalization":           "zero_mean_unit_var",  # per-batch z-score
    "training.loss_fn":                 "bce",   # multi-label -> independent BCE per class
    "training.early_stopping_patience": 5,       # stop after 5 epochs w/o val-AUROC gain
    # num_epochs is an upper bound only: early stopping nearly always fires
    # first, so 500 just means "don't cap before the model has plateaued".
    "training.num_epochs":              500,
    "model.embedding_dim":              256,
    "model.num_heads":                  4,
    "model.num_blocks":                 2,
    "model.fusion_type":                "layer",  # eq. 13 of the study (iterative mutual fusion)
    "model.dropout":                    0.5,
    "model.mask_ratio":                 0.2,      # fraction of seq positions eligible to mask
    "model.mask_prob":                  0.8,      # prob a given batch applies masking at all
    "model.pooling":                    "mean",   # mean-pool over the 59-step sequence

    "data.label_aggregation":           "lik_eq_100",
}

GEORGIA_DATA_PAPER_STRICT = {
    "data.dataset_type":                "georgia",
    # Georgia data paths come from config.yaml (set data.root there); they are
    # deliberately not overridden here, so there is one place to point at data.
    "data.georgia_split_strategy":      "paper_strict",
    "model.num_classes":                7,    # NSR/AF/IAVB/LBBB/RBBB/SB/STach
    "training.num_epochs":              20,   # Georgia budget from Section 4.7 of the study
    # patience 999 effectively disables early stopping: with only 20 epochs I
    # want every run to use the full fixed budget so depth comparisons are fair.
    "training.early_stopping_patience": 999,
}

SEEDS_THREE = [42, 123, 456]
SEEDS_TEN   = [42, 123, 456, 789, 999, 11, 13, 17, 23, 29]


# ── Phase A: single-axis search under correct labels ─────────────────────────

def phaseA_configs() -> list[dict]:
    """Build the single-axis hyperparameter search job list.

    Returns:
        list[dict]: one job spec per config, each with keys 'name' (unique
        run/dir name, 'A_*' prefixed), 'group' (all "A_search" so the summary
        groups them), and 'overrides' (the flat dotlist dict passed to
        run_training). ~35 configs total.
    """
    cfgs = []

    def add(name, **overrides):
        """Append one job: study recipe + these overrides, seed=42.

        'overrides' is merged after PAPER_PTBXL, so it wins on any shared key
        (that is how a single axis gets varied). 'training.seed' is appended
        last and is never overridden here, keeping phase A deterministic across
        the whole search.
        """
        cfgs.append({
            "name":      f"A_{name}",
            "group":     "A_search",
            "overrides": {**PAPER_PTBXL, **overrides, "training.seed": 42},
        })

    add("baseline")  # study recipe, single-seed -- the reference point every
    #                  other phase-A run is compared against.
    # normalization -- study default is zero_mean_unit_var; probe turning it off
    # entirely and a per-channel variant.
    for n in ("none", "per_channel"):
        add(f"norm_{n}",  **{"training.normalization": n})
    # scheduler -- the original study uses StepLR; here I trial cosine /
    # cosine+warmup / one-cycle / plateau-reduce alternatives. Each tuple is
    # (name-tag, the extra dotlist keys that scheduler needs, e.g.
    # warmup_epochs / lr_max).
    for tag, kv in [
        ("cosine",         {"training.lr_scheduler": "cosine"}),
        ("cosine_warmup5", {"training.lr_scheduler": "cosine_warmup", "training.warmup_epochs": 5}),
        ("cosine_warmup10",{"training.lr_scheduler": "cosine_warmup", "training.warmup_epochs": 10}),
        ("onecycle",       {"training.lr_scheduler": "onecycle", "training.lr_max_onecycle": 0.001}),
        ("reduce_lr",      {"training.lr_scheduler": "reduce_lr"}),
    ]:
        add(f"sched_{tag}", **kv)
    # optimizer / weight decay -- switch Adam -> AdamW and sweep the decoupled
    # weight-decay strength. (Plain Adam in the original study uses wd=0.)
    for wd in (0.001, 0.01, 0.1):
        # NOTE: the f-string interpolates the float verbatim, so the run name is
        # e.g. "A_adamw_wd0.001"
        add(f"adamw_wd{wd}", **{"training.optimizer": "adamw", "training.weight_decay": wd})
    # num_blocks -- depth sweep around the study's nb=2 (1 = shallower, up to 6).
    for nb in (1, 3, 4, 6):
        add(f"nb{nb}", **{"model.num_blocks": nb})
    # dropout -- the study uses 0.5; probe both lighter and (0.6) heavier.
    for dr in (0.1, 0.2, 0.3, 0.4, 0.6):
        add(f"drop{dr}", **{"model.dropout": dr})
    # loss -- alternatives to plain BCE for this class-imbalanced multi-label
    # setting: focal (down-weights easy negatives), label-smoothed BCE, and
    # class-frequency-weighted BCE. Each carries its own hyperparameters.
    for tag, kv in [
        ("focal",    {"training.loss_fn": "focal", "training.focal_gamma": 2.0, "training.focal_alpha": 0.25}),
        ("bce_smooth", {"training.loss_fn": "bce_smooth", "training.label_smoothing": 0.1}),
        ("bce_weighted", {"training.loss_fn": "bce_weighted"}),
    ]:
        add(f"loss_{tag}", **kv)
    # pooling -- the study mean-pools the sequence; try attention pooling and
    # last-step pooling instead.
    for p in ("attention", "last"):
        add(f"pool_{p}", **{"model.pooling": p})
    # embedding dim -- model width around the study's 256.
    for emb in (192, 320, 512):
        add(f"emb{emb}", **{"model.embedding_dim": emb})
    # heads -- sLSTM/mLSTM head count around the study's 4. Changing this (or
    # embedding_dim) triggers a sLSTM CUDA-kernel recompile on first use.
    for h in (2, 8):
        add(f"heads{h}", **{"model.num_heads": h})
    # fusion (varied at fixed depth nb=2) -- the three non-default fusion modes;
    # "layer" itself is already the baseline. Phase B repeats this 4-way with
    # multiple seeds for the actual Table 5 ablation.
    for f in ("slstm_only", "mlstm_only", "sequential"):
        add(f"fusion_{f}", **{"model.fusion_type": f})
    # mask params (the study's Tables 3/4 picks are already the default, so
    # sample around them) -- mask_ratio = fraction of positions eligible,
    # mask_prob = per-batch probability that masking is applied at all.
    for r in (0.1, 0.3, 0.5):
        add(f"mask_ratio{r}", **{"model.mask_ratio": r})
    for p in (0.6, 0.9, 1.0):
        add(f"mask_prob{p}", **{"model.mask_prob": p})

    return cfgs



def phaseB_configs() -> list[dict]:
    """Build the phase-B Table 5 fusion ablation job list.

    All four fusion variants at a fixed depth (nb=2), each repeated over
    SEEDS_THREE -> 4 x 3 = 12 runs. Fixing depth is the whole point: the earlier
    sweep9 version let depth vary alongside fusion, which confounds the
    comparison. Each fusion variant gets its own 'group' ('B_table5_<fusion>')
    so the summary reports a per-variant mean +/- std over the three seeds.

    Returns:
        list[dict]: 12 job specs (4 fusions x 3 seeds).
    """
    out = []
    for fusion in ("slstm_only", "mlstm_only", "sequential", "layer"):
        for s in SEEDS_THREE:
            out.append({
                "name": f"B_table5_{fusion}_nb2_s{s}",
                "group": f"B_table5_{fusion}",       # per-fusion group for mean +/- std
                "overrides": {**PAPER_PTBXL,
                              "model.fusion_type": fusion,
                              "model.num_blocks":  2,   # FIXED depth -- the fix vs sweep9
                              "training.seed":     s},
            })
    return out




def phaseC_configs() -> list[dict]:
    """Build the phase-C job list: the study's exact recipe x 10 seeds.

    Pure PAPER_PTBXL recipe, no perturbation, run across all ten SEEDS_TEN.
    This serves two downstream purposes: (1) a mean/std on the headline Table 6
    AUROC (how much the recipe's score moves with the seed), and (2) a 10-member
    checkpoint set that ensemble_eval.py later averages for the ensemble number.
    All ten share group 'C_paper_exact'.

    Returns:
        list[dict]: 10 job specs, one per seed.
    """
    return [
        {"name": f"C_paper_exact_s{s}", "group": "C_paper_exact",
         "overrides": {**PAPER_PTBXL, "training.seed": s}}
        for s in SEEDS_TEN
    ]



def phaseD_configs() -> list[dict]:
    """Build the phase-D Georgia job list under the study's recipe.

    Study recipe + Georgia overlay, swept over depth nb in {2,4,6} x SEEDS_THREE
    -> 9 runs, all under the strict group split (g1 test). Same content as
    sweep_mask_ablation phase 9, which was killed mid-run; redone here so Georgia lives in
    the same clean-labels sweep as everything else.

    Note the merge order: '{**PAPER_PTBXL, **GEORGIA_DATA_PAPER_STRICT, ...}'
    means the Georgia overlay overrides PTB-XL where they collide (dataset_type,
    num_epochs, patience, num_classes), then num_blocks/seed override on top.
    All nine share one group 'D_georgia_paper_recipe'.

    Returns:
        list[dict]: 9 job specs (3 depths x 3 seeds).
    """
    out = []
    for nb in (2, 4, 6):
        for s in SEEDS_THREE:
            out.append({
                "name": f"D_georgia_paper_nb{nb}_s{s}",
                "group": "D_georgia_paper_recipe",
                "overrides": {**PAPER_PTBXL, **GEORGIA_DATA_PAPER_STRICT,
                              "model.num_blocks": nb,
                              "training.seed":    s},
            })
    return out



def worker(gpu_id, work_queue, results, lock, start_time):
    """Thread body: drain the shared job queue, one training run at a time.

    A pool of these runs concurrently -- typically several per physical GPU (the
    '--gpus 0,0,0,1,1,1' pattern starts three workers pinned to GPU0 and three
    to GPU1). Each worker pops jobs until the queue is empty, then returns so its
    thread can be joined.

    Args:
        gpu_id: the CUDA device index this worker pins its subprocess to (via
            'CUDA_VISIBLE_DEVICES'). Multiple workers may share a gpu_id.
        work_queue: the shared 'queue.Queue' of job specs.
        results: shared list every worker appends finished result dicts to
            (guarded by 'lock').
        lock: the threading.Lock serializing writes to 'results' and the summary
            file.
        start_time: sweep wall-clock t0, used only for "+Xh" progress stamps.
    """
    while True:
        # get_nowait + catch-empty rather than a blocking get(): once the queue
        # is drained there is no more work coming (it is never refilled), so the
        # worker should exit.
        try: spec = work_queue.get_nowait()
        except queue.Empty: break
        _run_training(gpu_id, spec, results, lock, start_time)
        work_queue.task_done()


def _run_training(gpu_id, spec, results, lock, start_time):
    """Execute one training job as a subprocess and record its result.

    Resolves the per-run output directory, resumes if it already finished,
    otherwise launches 'python -m scripts.run_training' with this job's
    overrides on the assigned GPU, waits (with a hard timeout), then reads back
    the 'result.json' the training process wrote and appends an enriched copy to
    the shared 'results' list. Every exit path (success, no-output, timeout,
    unexpected exception) appends some dict, so no run goes unaccounted for in
    the summary.

    Args:
        gpu_id: CUDA device index for this run.
        spec: job dict with 'name' / 'group' / 'overrides' (as built by the
            phaseX_configs functions).
        results, lock, start_time: shared sweep state (see 'worker').

    Side effects:
        Creates 'SWEEP_DIR/<name>/' with 'train.log' (and the training
        subprocess writes 'result.json', 'checkpoints/', 'logs/' there), mutates
        'results' under 'lock', and rewrites 'summary.json'.
    """
    name, group, overrides = spec["name"], spec["group"], spec["overrides"]
    run_dir     = SWEEP_DIR / name
    result_file = run_dir / "result.json"
    log_file    = run_dir / "train.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    # A pre-existing result.json means this run already completed on an earlier
    # invocation; reuse it rather than burning GPU time. I still re-attach
    # name/group/overrides because the on-disk JSON carries only the metrics from
    # the training side, not this harness's bookkeeping fields.
    if result_file.exists():
        eh = (time.time() - start_time) / 3600
        print(f"[GPU{gpu_id}] SKIP  {name:<48} +{eh:.2f}h (done)", flush=True)
        with open(result_file) as f: result = json.load(f)
        result.update({"name": name, "group": group, "overrides": overrides})
        with lock: results.append(result)
        return

    # ── Build the subprocess command ─────────────────────────────────────────
    # The 'extra' keys are harness-injected output paths):
    # they tell the training process where to drop checkpoints, TB/log
    # dirs, the result.json this reads back.
    # training.result_file is the contract that turns run_training into a
    # metrics-producing job.
    extra = [
        f"training.checkpoint_dir={run_dir / 'checkpoints'}",
        f"training.log_dir={run_dir / 'logs'}",
        f"training.result_file={result_file}",
    ]
    # Flatten the overrides dict into OmegaConf dotlist tokens ("k=v") and append
    # the output-path tokens. run_training.py consumes these as its positional
    # 'overrides' argument.
    cmd = [PYTHON, "-m", "scripts.run_training"] \
          + [f"{k}={v}" for k, v in overrides.items()] + extra
    # Pin the run to one GPU by masking the rest. Inheriting os.environ keeps the
    # venv / CUDA paths; only CUDA_VISIBLE_DEVICES is overridden.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    eh = (time.time() - start_time) / 3600
    print(f"[GPU{gpu_id}] START {name:<48} +{eh:.2f}h", flush=True)
    t0 = time.time()
    try:
        # All subprocess stdout+stderr is redirected into the run's train.log.
        # I prepend the exact command + GPU + timestamp so a log is
        # self-describing and the run is reproducible from it alone.
        with open(log_file, "w") as lf:
            lf.write(f"CMD: {' '.join(str(c) for c in cmd)}\n")
            lf.write(f"GPU: {gpu_id}\nSTARTED: {datetime.now().isoformat()}\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf, cwd=str(PROJECT))

            # Hard 3-hour (10800 s) wall cap per run: a single training run that
            # lasts this long has almost certainly hung (kernel deadlock / stuck
            # dataloader), so kill it and let the except below record a timeout.
            # This is especially important as I do not get other notifications of potentially
            # killed (or finished runs) anymore.
            try: proc.wait(timeout=10800)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); raise  # re-raise -> outer timeout handler

        elapsed = time.time() - t0
        # Success is defined by the training side having produced result.json
        # (an exit code 0 with no file is still a failure as far as I'm
        # concerned).
        if result_file.exists():
            with open(result_file) as f: result = json.load(f)
            result.update({"name": name, "group": group, "overrides": overrides})
            au = result.get("test_auroc", float("nan"))  # NaN if key missing -> prints "nan"
            print(f"[GPU{gpu_id}] DONE  {name:<48} AUROC={au:.4f}  {elapsed/60:.1f}min",
                  flush=True)
        else:
            # Process returned but wrote no metrics -- record the return code so
            # the failure is diagnosable from summary.json without opening logs.
            result = {"name": name, "group": group, "overrides": overrides,
                      "error": "no result.json", "rc": proc.returncode}
            print(f"[GPU{gpu_id}] FAIL  {name:<48} rc={proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        # Reached via the re-raise above after we killed the hung process.
        result = {"name": name, "group": group, "overrides": overrides, "error": "timeout"}
        print(f"[GPU{gpu_id}] TIMEOUT {name}", flush=True)
    except Exception as e:
        # Catch-all so one broken job (e.g. JSON decode error, OOM in Popen)
        # never takes down the whole sweep -- it is logged as a failed run and
        # the worker moves on to the next job.
        result = {"name": name, "group": group, "overrides": overrides, "error": str(e)}
        print(f"[GPU{gpu_id}] ERROR {name}: {e}", flush=True)

    # Serialize the shared-state mutation: append this result and rewrite the
    # rolling summary so an external watcher always sees an up-to-date
    # leaderboard, even mid-sweep.
    with lock:
        results.append(result)
        _write_summary(results)


def _is_done(spec):
    """Return True if this job already has a result.json on disk (resume marker)."""
    return (SWEEP_DIR / spec["name"] / "result.json").exists()


def _write_summary(results):
    """Rewrite 'summary.json': a leaderboard of completed runs plus failures.

    Args:
        results: the shared list of result dicts accumulated so far.
    """
    # "completed" = has a test_auroc; sort descending so the best run is on top.
    done = sorted([r for r in results if "test_auroc" in r],
                  key=lambda r: r.get("test_auroc", 0), reverse=True)
    # "failed" = no metric but an error string was recorded.
    fail = [r for r in results if "test_auroc" not in r and "error" in r]
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump({
            "updated":   datetime.now().isoformat(),
            "training":  {"completed": len(done), "failed": len(fail), "runs": done},
            "all_failed": fail,
        }, f, indent=2)


def _print_group_summary(results, prefix):
    """Print a per-group leaderboard with mean/std AUROC for one phase.

    Args:
        results: shared list of result dicts.
        prefix: group-name prefix identifying this phase's runs.

    Note:
        std here is the population std (divides by N, not N-1) -- fine as a
        descriptive seed-spread, but not an unbiased sample estimate.
    """
    runs = [r for r in results if r.get("group", "").startswith(prefix)
            and "test_auroc" in r]
    if not runs: return  # nothing completed yet for this phase -- stay silent
    print(f"\n  Group prefix=~{prefix} ({len(runs)} runs):")
    for r in sorted(runs, key=lambda x: -x.get("test_auroc", 0)):
        print(f"    {r['name']:<48}  AUROC={r['test_auroc']:.4f}")
    aurocs = [r["test_auroc"] for r in runs]
    mean = sum(aurocs) / len(aurocs)
    # population std: sqrt(mean of the squared deviations).
    std  = (sum((x - mean) ** 2 for x in aurocs) / len(aurocs)) ** 0.5
    print(f"    Mean = {mean:.4f}  Std = {std:.4f}")


def run_phase(name, jobs, gpu_ids, results, lock, start_time, dry_run):
    """Run (or, in dry-run, just describe) all jobs of one phase to completion.

    Args:
        name: human-readable phase name, for the log header.
        jobs: list of job specs from a phaseX_configs builder.
        gpu_ids: list of GPU indices; len = number of worker threads. Repeated
            indices = multiple workers on the same physical GPU.
        results, lock, start_time: shared sweep state.
        dry_run: if True, print the plan and return without launching anything.
    """
    if not jobs:
        print(f"\n[sweep_main] Phase '{name}': no jobs.")
        return
    pending = [j for j in jobs if not _is_done(j)]
    print(f"\n[sweep_main] Phase '{name}': {len(pending)}/{len(jobs)} to run "
          f"({len(jobs) - len(pending)} done).", flush=True)
    if dry_run:
        # Show, per job, only the keys that actually deviate from PAPER_PTBXL
        # (a key absent from PAPER_PTBXL, e.g. a Georgia/seed key, also counts
        # as a diff). This makes "what is this run actually testing?" obvious
        # without dumping the whole ~20-key recipe each time.
        for j in jobs:
            diff = {k: v for k, v in j["overrides"].items()
                    if k not in PAPER_PTBXL or PAPER_PTBXL.get(k) != v}
            print(f"  TRAIN {j['name']:<48}  diff_from_paper={diff}")
        return
    # Single shared queue; workers pull from it. Done jobs are deliberately NOT
    # filtered out here -- _run_training short-circuits them via the resume path.
    wq = queue.Queue()
    for j in jobs: wq.put(j)
    # One daemon thread per gpu_id slot. daemon=True so a Ctrl-C on the main
    # thread doesn't hang on stuck workers. I join() all of them below, which is
    # what makes phases run strictly sequentially (phase A fully finishes before
    # phase B starts).
    threads = [threading.Thread(target=worker,
                                 args=(g, wq, results, lock, start_time),
                                 daemon=True, name=f"GPU{g}-w{i}")
               for i, g in enumerate(gpu_ids)]
    for t in threads: t.start()
    for t in threads: t.join()


# Phase registry: CLI letter -> (phase_name, job-builder fn, group prefix for
# the post-phase summary).
PHASES = {
    "A": ("A_search",                  phaseA_configs, "A_search"),
    "B": ("B_table5_fixed_depth",      phaseB_configs, "B_table5_"),
    "C": ("C_paper_exact_x10seeds",    phaseC_configs, "C_paper_exact"),
    "D": ("D_georgia_paper_recipe",    phaseD_configs, "D_georgia_paper_recipe"),
}


def main():
    """CLI entry point: parse args, run the requested phases, print summaries.

    Flags:
        --gpus    Comma list of GPU indices; one worker per element (repeats =
                  multiple workers per GPU). Default "0,0,0,1,1,1" = 3+3.
        --dry-run Plan only: print each run's diff from the study recipe, train
                  nothing.
        --phases  Comma list of phase letters to run, in the given order.
                  Unknown letters are skipped with a warning. Default "A,B,C,D".
    """
    p = argparse.ArgumentParser(description="sweep_main: clean rebuild under correct labels")
    p.add_argument("--gpus",    default="0,0,0,1,1,1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--phases",  default="A,B,C,D")
    args = p.parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]   # "0,0,1" -> [0,0,1]
    phases  = [s.strip() for s in args.phases.split(",")]  # tolerate "A, B"
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"sweep_main (clean-labels rebuild)  |  phases={phases}  |  GPUs: {args.gpus}")
    print(f"Results: {SWEEP_DIR}")
    # Report the worker count per physical GPU (Counter over the gpu_ids list).
    counts = Counter(gpu_ids)
    for gid, n in sorted(counts.items()):
        print(f"  GPU{gid}: {n} workers")
    print('=' * 72)

    results: list = []
    lock = threading.Lock()
    start_time = time.time()

    # Phases run strictly in the order given on --phases (each run_phase joins
    # its workers before returning), and the per-phase summary is printed
    # immediately after that phase finishes.
    for ph in phases:
        if ph not in PHASES:
            print(f"[sweep_main] Unknown phase {ph}, skipping.")
            continue
        phase_name, builder, group_substr = PHASES[ph]

        run_phase(phase_name, builder(),
                  gpu_ids, results, lock, start_time, args.dry_run)

        if not args.dry_run:
            _print_group_summary(results, group_substr)

    if not args.dry_run:
        eh = (time.time() - start_time) / 3600
        n_train = sum(1 for r in results if "test_auroc" in r)
        n_fail  = sum(1 for r in results if "test_auroc" not in r and "error" in r)
        print(f"\nDONE  {eh:.2f}h  |  {n_train} runs  |  {n_fail} failures")

        # NOTE: emit these next-step commands with the actual best-run paths
        #       (e.g. read summary.json and print the top phase-C run) so they
        #       can be copy-pasted rather than hand-filled.
        print("\nNext steps:")
        print("  - threshold_tune.py on top phase-A and phase-C checkpoints")
        print("  - ensemble_eval.py over phase-C ten seeds")
        print("  - bootstrap_ci.py on the ensemble + best individual run")


if __name__ == "__main__":
    main()
