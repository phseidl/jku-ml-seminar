"""
scripts/sweep_leave_one_out.py
===================================
A true leave-one-out ablation on the corrected (results_matching) recipe.

Usage:
    python scripts/sweep_leave_one_out.py --gpus 1,1,1
    python scripts/sweep_leave_one_out.py --dry-run

"""

import argparse
import json
import os
import queue       # thread-safe job queue: workers pull jobs with get_nowait()
import subprocess  # each training run is an isolated child process
import sys
import threading   # one worker thread per --gpus entry; a Lock guards 'results'
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT   = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT / "results" / "sweep_leave_one_out"
PYTHON    = sys.executable

CORRECTED_FULL = {
    "training.optimizer":               "adam",
    "training.learning_rate":           2e-4,
    "training.lr_scheduler":            "step",   # StepLR -- held fixed in all variants
    "training.step_lr_step_size":       2,        # decay every 2 epochs ...
    "training.step_lr_gamma":           0.8,      # ... multiply LR by 0.8 (original study StepLR(2, 0.8))
    "training.weight_decay":            0.0,
    "training.loss_fn":                 "bce",    # multi-label binary cross-entropy
    "training.early_stopping_patience": 5,        # stop after 5 epochs without val-AUROC gain
    "training.num_epochs":              500,      # ceiling; early stopping ends runs far sooner
    "model.embedding_dim":              512,      # ABLATED axis (original-study value 256)
    "model.num_heads":                  4,
    "model.num_blocks":                 4,        # ABLATED axis (original-study value 2)
    "model.fusion_type":                "layer",  # original study eq. 13 iterative mutual refinement
    "model.dropout":                    0.1,      # ABLATED axis (original-study value 0.5)
    "model.mask_ratio":                 0.2,
    "model.mask_prob":                  0.8,
    "model.pooling":                    "mean",
    "data.label_aggregation":           "lik_eq_100",  # likelihood==100 only -> high-confidence labels
}

# Leave-one-out grid: each variant overrides EXACTLY ONE key of CORRECTED_FULL
# back to its value in Section 4.3 of the original study (merged on top via **
# spread in configs()).
ABLATIONS = {
    # Each entry: only the differences from CORRECTED_FULL (revert one change
    # to its original-study value).
    "revert_dropout": {"model.dropout":      0.5},  # 0.1 -> 0.5 (original-study dropout)
    "revert_emb":     {"model.embedding_dim": 256}, # 512 -> 256 (original-study embedding)
    "revert_depth":   {"model.num_blocks":    2},   # 4 -> 2 (original-study diagram depth)
}

SEEDS = [42, 123, 456]

# Reference numbers for the printed delta summary.
CORRECTED_MEAN     = 0.9063   # results_matching 10-seed mean
PAPER_FAITHFUL_MEAN = 0.8861  # original-study-recipe 10-seed mean


def configs() -> list[dict]:
    """Expand the ablation x seed grid into a flat list of job specs.

    Builds one spec per (variant, seed) pair -- 3 variants x 3 seeds = 9 specs.
    Each spec's 'overrides' is the full recipe handed to run_training: the
    corrected recipe, with this variant's single reverted key spread on top,
    plus the per-run seed.

    Returns:
        A list of dicts, each with keys:
          * 'name'      -- unique run id, e.g. "revert_depth_s42".
          * 'group'     -- the ablation name, used to group seeds for the
                           per-variant mean/std in main().
          * 'outdir'    -- results/sweep_leave_one_out/<group>/s<seed>/
                           where this run's checkpoints, logs and result.json land.
          * 'overrides' -- the complete OmegaConf dotlist recipe for the run.
    """
    out = []
    for ablation_name, override in ABLATIONS.items():
        for s in SEEDS:
            # Dict-merge order matters: CORRECTED_FULL first, then the variant
            # delta (so the reverted key wins), then the seed. Later keys
            # overwrite earlier ones, which is exactly the leave-one-out intent.
            recipe = {**CORRECTED_FULL, **override, "training.seed": s}
            out.append({
                "name":   f"{ablation_name}_s{s}",
                "group":  ablation_name,
                "outdir": SWEEP_DIR / ablation_name / f"s{s}",
                "overrides": recipe,
            })
    return out

# NOTE: worker / _run_training / _write_summary are copy-pasted near-verbatim
# across every sweep*.py harness. They differ only in SWEEP_DIR, the timeout,
# and which path overrides are appended. Lifting them into a shared
# 'scripts/_sweep_runner.py' (parametrized by sweep dir + extra-overrides
# builder) would remove the duplication without changing any run's behavior.

def worker(gpu_id, work_queue, results, lock, start_time):
    """Thread body: drain the shared queue, running one job at a time on 'gpu_id'.

    Args:
        gpu_id:     CUDA device index this worker pins its runs to (passed
                    through as CUDA_VISIBLE_DEVICES in the child env).
        work_queue: shared 'queue.Queue' of job spec dicts from configs().
        results:    shared list every worker appends results to (guarded by lock).
        lock:       'threading.Lock' serializing appends to 'results' and the
                    summary write.
        start_time: 'time.time()' at sweep launch, for the elapsed-hours stamps.
    """
    while True:
        try:
            spec = work_queue.get_nowait()
        except queue.Empty:
            break
        _run_training(gpu_id, spec, results, lock, start_time)
        work_queue.task_done()


def _run_training(gpu_id, spec, results, lock, start_time):
    """Run (or resume) one training job as a subprocess and record its result.

    Args:
        gpu_id:     CUDA device index for this run (set via CUDA_VISIBLE_DEVICES).
        spec:       one job dict from configs().
        results:    shared results list (append guarded by lock).
        lock:       lock serializing the results append + summary write.
        start_time: sweep start timestamp, for the +Nh elapsed stamps.
    """
    name, group, overrides, outdir = (
        spec["name"], spec["group"], spec["overrides"], Path(spec["outdir"])
    )
    result_file = outdir / "result.json"
    log_file    = outdir / "train.log"
    outdir.mkdir(parents=True, exist_ok=True)

    if result_file.exists():
        eh = (time.time() - start_time) / 3600
        print(f"[GPU{gpu_id}] SKIP  {name:<40} +{eh:.2f}h (done)", flush=True)
        with open(result_file) as f:
            result = json.load(f)

        result.update({"name": name, "group": group, "overrides": overrides})
        with lock:
            results.append(result)
        return


    extra = [
        f"training.checkpoint_dir={outdir / 'checkpoints'}",
        f"training.log_dir={outdir / 'logs'}",
        f"training.result_file={result_file}",
    ]
    # Build the argv: interpreter, module, the recipe as key=value tokens, then
    # the path overrides. run_training merges these via OmegaConf.from_dotlist.
    cmd = [PYTHON, "-m", "scripts.run_training"] \
          + [f"{k}={v}" for k, v in overrides.items()] + extra
    # Child env: copy ours, then pin the GPU (CUDA_VISIBLE_DEVICES makes the
    # child see only this one device as cuda:0) and enable expandable CUDA
    # segments to cut fragmentation/OOM when several workers share a GPU.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    eh = (time.time() - start_time) / 3600
    print(f"[GPU{gpu_id}] START {name:<40} +{eh:.2f}h", flush=True)
    t0 = time.time()  # per-run wall clock, reported as minutes on completion
    try:
        with open(log_file, "w") as lf:
            lf.write(f"CMD: {' '.join(str(c) for c in cmd)}\n")
            lf.write(f"GPU: {gpu_id}\nSTARTED: {datetime.now().isoformat()}\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf,
                                    cwd=str(PROJECT))
            proc.wait(timeout=10800)

        elapsed = time.time() - t0

        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            result.update({"name": name, "group": group, "overrides": overrides})
            au = result.get("test_auroc", float("nan"))
            print(f"[GPU{gpu_id}] DONE  {name:<40} AUROC={au:.4f}  {elapsed/60:.1f}min",
                  flush=True)
        else:
            result = {"name": name, "group": group, "overrides": overrides,
                      "error": "no result.json", "rc": proc.returncode}
            print(f"[GPU{gpu_id}] FAIL  {name:<40} rc={proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        # The 3h cap fired: kill the child and reap it so no zombie or pinned
        # GPU memory is left behind, then record a timeout failure.
        proc.kill(); proc.wait()
        result = {"name": name, "group": group, "overrides": overrides, "error": "timeout"}
        print(f"[GPU{gpu_id}] TIMEOUT {name}", flush=True)
    except Exception as e:
        # Catch-all so one bad job (e.g. log file unwritable, OOM at launch)
        # never kills the worker thread or aborts the rest of the sweep.
        # NOTE: this bare Exception will not catch a TimeoutExpired (handled
        # above) but swallows everything else; the message is stored for triage.
        result = {"name": name, "group": group, "overrides": overrides, "error": str(e)}
        print(f"[GPU{gpu_id}] ERROR {name}: {e}", flush=True)

    with lock:
        results.append(result)
        _write_summary(results)


def _write_summary(results):
    """Rewrite summary.json from the current results -- the rolling snapshot.

    Args:
        results: the shared results list (caller already holds the lock).
    """
    # "completed" = produced a test_auroc; sort descending so the best run is
    # first in the file. default=0 guards the rare partial dict missing the key.
    done = sorted([r for r in results if "test_auroc" in r],
                  key=lambda r: r.get("test_auroc", 0), reverse=True)
    # "failed" = no metric but an error string was recorded (skip/done go to 'done').
    fail = [r for r in results if "test_auroc" not in r and "error" in r]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    # Overwrite (not append) each time: the snapshot is meant to be the latest
    # full state, and indent=2 keeps it readable for a quick spot check.
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump({
            "updated":   datetime.now().isoformat(),
            "training":  {"completed": len(done), "failed": len(fail), "runs": done},
            "all_failed": fail,
        }, f, indent=2)


def main():
    """CLI entry point: plan the grid, then run it across GPU workers (or dry-run).
    """
    p = argparse.ArgumentParser()
    # --gpus: comma-separated CUDA indices; ONE worker thread per entry, so a
    # repeated index (e.g. "1,1,1") means several workers share that one GPU.
    p.add_argument("--gpus",    default="1,1,1")
    # --dry-run: print the planned jobs and their diffs, launch nothing.
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    # "1,1,1" -> [1, 1, 1]; the length is the worker count, the values are the
    # device each worker pins to.
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    jobs = configs()
    print(f"\n{'=' * 72}")
    print(f"sweep_leave_one_out  |  {len(jobs)} jobs  |  GPUs: {args.gpus}")
    print(f"Results: {SWEEP_DIR}")
    # Counter collapses the GPU list into {device: worker_count} so the banner
    # shows how many workers land on each physical GPU.
    counts = Counter(gpu_ids)
    for gid, n in sorted(counts.items()):
        print(f"  GPU{gid}: {n} workers")
    print('=' * 72)

    if args.dry_run:
        # Show exactly what each job changes relative to the corrected recipe.
        # The diff should always be the single reverted axis plus training.seed
        # -- a quick eyeball check that the leave-one-out grid is right before
        # committing GPU hours.
        for j in jobs:
            diff = {k: v for k, v in j["overrides"].items()
                    if k not in CORRECTED_FULL or CORRECTED_FULL.get(k) != v}
            print(f"  {j['name']:<40}  diff_from_corrected_full={diff}")
        return

    # Shared state across worker threads: one results list + one lock guarding
    # both the append and the summary write. start_time anchors every elapsed
    # stamp printed during the run.
    results: list = []
    lock = threading.Lock()
    start_time = time.time()
    # Fill the queue up front with all 9 jobs; the workers then self-serve. This
    # is the classic thread-pool-over-a-queue pattern (a worker pulls the next
    # job whenever its GPU frees up), so jobs are not statically pinned per
    # worker.
    wq: queue.Queue = queue.Queue()
    for j in jobs:
        wq.put(j)
    # One thread per --gpus entry. daemon=True so a Ctrl-C on the main thread
    # does not hang on stuck workers; 'i' only disambiguates the thread name
    # when several workers share a GPU index.
    threads = [threading.Thread(target=worker,
                                 args=(g, wq, results, lock, start_time),
                                 daemon=True, name=f"GPU{g}-w{i}")
               for i, g in enumerate(gpu_ids)]
    for t in threads: t.start()
    for t in threads: t.join()  # block until every job has been drained

    eh = (time.time() - start_time) / 3600
    # Final tally uses the same test_auroc-presence rule as _write_summary, so
    # the printed counts match summary.json exactly.
    n_train = sum(1 for r in results if "test_auroc" in r)
    n_fail  = sum(1 for r in results if "test_auroc" not in r and "error" in r)
    print(f"\nDONE  {eh:.2f}h  |  {n_train} runs  |  {n_fail} failures")

    # Per-ablation summary, laid out so it reads cleanly in the log.
    import statistics

    # Bucket each completed run's test_auroc under its variant name, so each
    # group holds the (up to) 3 seed scores for that reverted axis.
    by_group: dict[str, list[float]] = {}
    for r in results:
        if "test_auroc" in r:
            by_group.setdefault(r["group"], []).append(r["test_auroc"])
    print("\nTrue leave-one-out ablation on the CORRECTED recipe "
          "(StepLR + dropout=0.1 + emb=512 + nb=4):")
    print(f"  Reference: corrected 10-seed mean = {CORRECTED_MEAN:.4f}; "
          f"paper-faithful 10-seed mean = {PAPER_FAITHFUL_MEAN:.4f}")
    print(f"  {'Variant':<16}  {'Mean':>7}  {'Std':>7}  "
          f"{'Δ vs corrected':>14}  {'Δ vs paper':>11}  seeds")

    # Print order is depth, emb, dropout -- the order the report's table uses,
    # which differs from ABLATIONS' insertion order, so it is spelled out here.
    for v in ("revert_depth", "revert_emb", "revert_dropout"):
        a = sorted(by_group.get(v, []))  # sorted so the printed seed list is ascending

        if a:  # skip a variant entirely if none of its seeds completed

            # pstdev = POPULATION std (divides by n, not n-1): the 3 seeds ARE
            # the whole set I ran, not a sample of a larger pool, so the
            # population estimator is the right one here.
            m = sum(a)/len(a); sd = statistics.pstdev(a)

            # The two deltas show how reverting this one axis moves the mean off
            # the corrected headline and relative to the original-study-faithful
            # baseline; +.4f forces an explicit sign so a gain or regression
            # reads at a glance.
            print(f"  {v:<16}  {m:.4f}  {sd:.4f}  {m - CORRECTED_MEAN:+.4f}        "
                  f"{m - PAPER_FAITHFUL_MEAN:+.4f}     {[f'{x:.4f}' for x in a]}")

if __name__ == "__main__":
    main()
