"""
scripts/sweep_results_matching_georgia.py
====================================

Trains the 'results_matching' recipe (Section 4.3 of the original
study, plus dropout=0.1, emb=512, nb=4, but KEEPING the original study's
StepLR schedule) on the Georgia challenge under the paper-strict split.
The point is to run Georgia through the same recipe that gives our PTB-XL
headline, so the two numbers are comparable rather than the
product of two different recipes.

Usage:
    python scripts/sweep_results_matching_georgia.py --gpus 0,0,1,1
    python scripts/sweep_results_matching_georgia.py --dry-run   # print plan only
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

SWEEP_DIR = PROJECT / "results" / "sweep_results_matching_georgia"

# Reuse the *exact* interpreter running this harness for the child trainer
# processes (sys.executable is the venv python), so the subprocess inherits the
# same environment/packages rather than whatever bare 'python' resolves to.
PYTHON    = sys.executable

# results_matching recipe = Section 4.3 of the original study + 3 hyperparameter
# changes:
#   dropout 0.5 -> 0.1
#   embedding_dim 256 -> 512
#   num_blocks 2 -> 4 (varied here as the swept axis)

MINUS_SCHED = {
    "training.optimizer":               "adam",
    "training.learning_rate":           2e-4,                # LR from Section 4.3 of the original study
    "training.lr_scheduler":            "step",              # StepLR -- the KEPT piece
    "training.step_lr_step_size":       2,                   # decay every 2 epochs
    "training.step_lr_gamma":           0.8,                 # x0.8 per decay step
    "training.weight_decay":            0.0,
    "training.loss_fn":                 "bce",               # multi-label -> per-class BCE
    "training.early_stopping_patience": 999,                 # Section 4.7 of the original study
    "training.num_epochs":              20,                  # Section 4.7 of the original study
    # The three corrected-recipe deltas vs the Section 4.3 values of the original study:
    "model.embedding_dim":              512,                 # changed from 256
    "model.num_heads":                  4,                   # 512/4 = 128-dim heads
    "model.fusion_type":                "layer",             # eq. 13 of the original study (iterative fuse)
    "model.dropout":                    0.1,                 # changed from 0.5
    "model.mask_ratio":                 0.2,                 # masked-token fraction (train aug)
    "model.mask_prob":                  0.8,                 # prob an individual sample gets masked at all (per-sample Bernoulli draw)
    "model.pooling":                    "mean",              # mean-pool over the 59-step sequence
    # NOTE: model.num_blocks is deliberately absent here -- it is the swept axis
    # and is injected per-config in configs() so one recipe dict can serve all
    # three depths.
}

GEORGIA_PAPER_STRICT = {
    "data.dataset_type":            "georgia",
    "data.georgia_split_strategy":  "paper_strict",
    # Georgia uses 7 SNOMED-CT classes (NSR/AF/IAVB/LBBB/RBBB/SB/STach) vs
    # PTB-XL's 5 superclasses, so the classifier head width changes with it.
    "model.num_classes":            7,
}

SEEDS_THREE = [42, 123, 456]
SEEDS_TEN   = [42, 123, 456, 789, 999, 11, 13, 17, 23, 29]


def configs() -> list[dict]:
    """Build the flat list of run specs for the whole sweep.

    Expands the depth x seed grid -- '(nb=2, 3 seeds)', '(nb=4, 3 seeds)',
    '(nb=6, 10 seeds)' -- into one dict per training run. Each spec is a
    self-contained job for a worker: a display 'name', the 'group' it rolls
    up into for the per-depth summary, the 'outdir' for its artifacts, and the
    fully merged 'overrides' dict handed to the trainer.

    Returns:
        list[dict]: 16 run specs (3 + 3 + 10), in (nb, seed) order. The order is
        only the order they are *enqueued*; actual execution order is whatever
        the worker threads dequeue, so results must never be index-matched to
        this list.

    """
    out = []
    # One (num_blocks, seed-family) tuple per depth; the deepest depth gets the
    # ten-seed family because it is the headline.
    for nb, seeds in [(2, SEEDS_THREE), (4, SEEDS_THREE), (6, SEEDS_TEN)]:
        for s in seeds:
            out.append({
                "name":   f"nb{nb}_s{s}",          # human-readable run id, e.g. "nb6_s17"
                "group":  f"nb{nb}",               # rollup key for the per-depth summary
                "outdir": SWEEP_DIR / f"nb{nb}" / f"s{s}",   # one dir per run
                # Dict-unpacking merge: base Georgia/recipe overrides, then the
                # two per-run axes. Dotted keys land directly in the trainer CLI.
                "overrides": {**GEORGIA_PAPER_STRICT, **MINUS_SCHED,
                              "model.num_blocks": nb,
                              "training.seed":    s},
            })
    return out


def worker(gpu_id, work_queue, results, lock, start_time):
    """Thread body: drain the shared queue, training one run at a time.

    Args:
        gpu_id: Physical GPU index this worker is pinned to (passed through to
            the child process via 'CUDA_VISIBLE_DEVICES'). Multiple workers may
            share a GPU.
        work_queue: Shared 'queue.Queue' of run specs from 'configs()'.
        results: Shared list every worker appends finished result dicts to.
        lock: 'threading.Lock' serializing writes to 'results' and the
            rolling summary file.
        start_time: Wall-clock epoch seconds when the sweep began, used for the
            '+Xh' elapsed stamps in the progress log.

    Loops until the queue is empty. 'get_nowait' (rather than a blocking
    'get') is the termination signal.
    """
    while True:
        try:
            # Non-blocking pop: an empty queue means "no work left" -> exit.
            spec = work_queue.get_nowait()
        except queue.Empty:
            break
        _run_training(gpu_id, spec, results, lock, start_time)
        work_queue.task_done()


def _run_training(gpu_id, spec, results, lock, start_time):
    """Run (or skip) one training job and record its result.

    Launches 'scripts.run_training' as a child process with the spec's
    overrides, waits for it, then reads back the 'result.json' the trainer
    wrote. Every exit path -- success, missing output, timeout, or unexpected
    exception -- ends by appending a result dict to the shared 'results' list,
    so the sweep accounting stays complete even when individual runs fail.

    Args:
        gpu_id: GPU index for 'CUDA_VISIBLE_DEVICES' (string-formatted).
        spec: One dict from 'configs()' (name/group/overrides/outdir).
        results: Shared results list (appended under 'lock').
        lock: Lock guarding 'results' and the rolling summary.
        start_time: Sweep start (epoch seconds) for the '+Xh' stamps.
    """
    # Unpack the spec.
    name, group, overrides, outdir = (
        spec["name"], spec["group"], spec["overrides"], Path(spec["outdir"])
    )
    result_file = outdir / "result.json"   # the trainer's metrics contract file
    log_file    = outdir / "train.log"     # this harness's own stdout/stderr capture
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Idempotent resume: a completed run is reused, not retrained. ---
    # This makes the whole sweep crash-safe -- rerun the command and only the
    # missing seeds get recomputed.
    if result_file.exists():
        eh = (time.time() - start_time) / 3600
        print(f"[GPU{gpu_id}] SKIP  {name:<20} +{eh:.2f}h (done)", flush=True)
        with open(result_file) as f:
            result = json.load(f)
        # Stamp the harness-level fields back on -- the trainer's result.json
        # knows nothing about this sweep's name/group/override provenance.
        result.update({"name": name, "group": group, "overrides": overrides})
        with lock:
            results.append(result)
        return

    # --- Per-run output routing. ---
    # These three keys redirect the trainer's artifacts into THIS run's outdir
    # (otherwise every run would collide on config.yaml's default paths).
    extra = [
        f"training.checkpoint_dir={outdir / 'checkpoints'}",
        f"training.log_dir={outdir / 'logs'}",
        f"training.result_file={result_file}",
    ]
    # Build the argv: interpreter, the trainer module, every override as a single
    # 'key=value' token (OmegaConf dotlist), then the per-run output routing.
    cmd = [PYTHON, "-m", "scripts.run_training"] \
          + [f"{k}={v}" for k, v in overrides.items()] + extra

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    eh = (time.time() - start_time) / 3600
    print(f"[GPU{gpu_id}] START {name:<20} +{eh:.2f}h", flush=True)
    t0 = time.time()   # per-run timer (separate from the sweep-wide start_time)
    try:
        # Tee both child streams into the run's train.log. The header records the
        # exact command + GPU + start time so a failed run is reproducible from
        # the log alone.
        with open(log_file, "w") as lf:
            lf.write(f"CMD: {' '.join(str(c) for c in cmd)}\n")
            lf.write(f"GPU: {gpu_id}\nSTARTED: {datetime.now().isoformat()}\n\n")
            lf.flush()
            # stdout AND stderr -> same file handle so interleaving is preserved.
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf,
                                    cwd=str(PROJECT))
            # 10800 s = 3 h hard cap per run; a single 20-epoch Georgia run is
            # minutes, so this only fires on a hung/wedged process.
            proc.wait(timeout=10800)

        elapsed = time.time() - t0
        # Success is defined by the trainer having written result.json -- a clean
        # exit code alone is not trusted, the metrics file is the real signal.
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
            result.update({"name": name, "group": group, "overrides": overrides})
            # NaN default keeps the .4f format string from crashing if a run
            # somehow wrote a result.json without test_auroc.
            au = result.get("test_auroc", float("nan"))
            print(f"[GPU{gpu_id}] DONE  {name:<20} AUROC={au:.4f}  {elapsed/60:.1f}min",
                  flush=True)
        else:
            # Process returned but produced no metrics -> treat as a failure and
            # capture the return code for triage.
            result = {"name": name, "group": group, "overrides": overrides,
                      "error": "no result.json", "rc": proc.returncode}
            print(f"[GPU{gpu_id}] FAIL  {name:<20} rc={proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        # Kill the wedged child and reap it so it does not become a zombie, then
        # record a timeout sentinel result.
        proc.kill(); proc.wait()
        result = {"name": name, "group": group, "overrides": overrides, "error": "timeout"}
        print(f"[GPU{gpu_id}] TIMEOUT {name}", flush=True)
    except Exception as e:
        # Catch-all so one broken run (e.g. disk full, bad override) never takes
        # down the worker thread or the rest of the sweep. The message is stored
        # verbatim for debugging.
        result = {"name": name, "group": group, "overrides": overrides, "error": str(e)}
        print(f"[GPU{gpu_id}] ERROR {name}: {e}", flush=True)

    with lock:
        results.append(result)
        _write_summary(results)


def _write_summary(results):
    """Rewrite 'summary.json' from the current results snapshot.

    Called under 'lock' after every run finishes, so the file is a live,
    monotonically-growing progress view: a sweep watcher can tail it to see how
    many runs have completed/failed without parsing per-run logs.

    Args:
        results: The shared results list as it stands now (a mix of completed
            runs carrying 'test_auroc' and failures carrying 'error').

    A 'completed' run is one with a 'test_auroc' key; a 'failed' run is one
    without it but with an 'error' key.
    """
    # Completed runs, sorted best-AUROC-first so the leaderboard is readable.
    done = sorted([r for r in results if "test_auroc" in r],
                  key=lambda r: r.get("test_auroc", 0), reverse=True)
    # Failures: anything that did not produce a metric but recorded an error.
    fail = [r for r in results if "test_auroc" not in r and "error" in r]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump({
            "updated":   datetime.now().isoformat(),   # wall-clock of this rewrite
            "training":  {"completed": len(done), "failed": len(fail), "runs": done},
            "all_failed": fail,
        }, f, indent=2)
    # NOTE: this overwrites in place rather than writing to a temp file and
    # renaming, so a crash mid-write could leave a truncated summary.json. The
    # per-run result.json files are the source of truth, so that is acceptable
    # here; a tmp+rename would make summary.json crash-atomic if it ever mattered.


def post_ensemble_and_ci_nb6():
    """Ensemble + 2 bootstrap CIs on the nb=6 10-seed set (the Georgia headline).

    """
    nb6_dir = SWEEP_DIR / "nb6"
    # Glob + sort the per-seed checkpoints. sorted() gives a stable member order
    # so the ensemble is reproducible; ensemble_eval also asserts a consistent
    # label order across members, so deterministic ordering here matters.
    ckpts = sorted(nb6_dir.glob("s*/checkpoints/best.pt"))
    if len(ckpts) < 10:
        # Warn but DO NOT abort: a partial ensemble (e.g. 8/10 seeds trained) is
        # still informative and the report can note the reduced N. ensemble_eval
        # itself works with any N >= 1.
        print(f"[ms-georgia] WARNING: only {len(ckpts)} of 10 nb=6 ckpts found",
              flush=True)

    # --- Job 1: build the ensemble. ---
    print(f"[ms-georgia] ensembling {len(ckpts)} nb=6 ckpts", flush=True)
    cmd = [PYTHON, "-m", "scripts.ensemble_eval",
           "--checkpoints", *[str(p) for p in ckpts],   # splat all member paths
           "--out", str(SWEEP_DIR / "ensemble_results_matching_georgia_nb6.json")]
    # Pin post-processing inference to GPU 0; the sweep's training threads have
    # already joined by the time main() calls this, so GPU 0 is free.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    log = PROJECT / "logs" / "ensemble_results_matching_georgia.log"
    with open(log, "w") as lf:
        # stderr folded into stdout so the single log holds the full child output.
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT)).returncode
    print(f"[ms-georgia] ensemble done rc={rc}", flush=True)
    if rc != 0:
        # The CI steps depend on the ensemble's npy sidecars; if the ensemble
        # failed those do not exist, so abort the whole post-processing block.
        return

    # --- Job 2: bootstrap CI on the ENSEMBLE scores. ---
    print("[ms-georgia] bootstrap CI on ensemble", flush=True)
    cmd_ci = [PYTHON, "-m", "scripts.bootstrap_ci",
              "--labels", str(SWEEP_DIR / "ensemble_results_matching_georgia_nb6_labels.npy"),
              "--scores", str(SWEEP_DIR / "ensemble_results_matching_georgia_nb6_scores.npy"),
              # Georgia's 7 SNOMED-CT class names, in the model's output order, so
              # the CI JSON's per-class entries are labeled correctly.
              "--class-names", "NSR", "AF", "IAVB", "LBBB", "RBBB", "SB", "STach",
              "--out", str(SWEEP_DIR / "ci_ensemble_results_matching_georgia_nb6.json")]
    log_ci = PROJECT / "logs" / "ci_ensemble_results_matching_georgia.log"
    with open(log_ci, "w") as lf:
        # NOTE: no env= -- this inherits the parent environment (CPU-only
        # numpy); the score path is intentionally device-agnostic.
        rc = subprocess.run(cmd_ci, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT)).returncode
    print(f"[ms-georgia] ensemble bootstrap CI done rc={rc}", flush=True)

    # --- Job 3: bootstrap CI on the single BEST individual nb=6 run. ---
    # Find the highest-AUROC seed by reading each result.json. This re-loads the
    # checkpoint and re-runs inference (the --checkpoint path of bootstrap_ci),
    # so it is slower than job 2 but gives the best-single-run error bar.
    best = max(nb6_dir.glob("s*/result.json"),
               key=lambda p: json.load(open(p))["test_auroc"])
    # Derive the checkpoint path from the result.json path by string-swapping the
    # filename within the same run directory.
    # NOTE: str.replace would also rewrite any earlier "result.json" substring in
    #   the path; in practice the run dirs never contain that token elsewhere, so
    #   it is safe.
    best_ckpt = str(best).replace("result.json", "checkpoints/best.pt")
    print(f"[ms-georgia] best nb=6 ckpt: {best_ckpt}", flush=True)
    cmd_best = [PYTHON, "-m", "scripts.bootstrap_ci",
                "--checkpoint", best_ckpt,
                "--out", str(SWEEP_DIR / "ci_results_matching_georgia_best.json")]
    log_best = PROJECT / "logs" / "ci_results_matching_georgia_best.log"
    with open(log_best, "w") as lf:
        # env= restored: the --checkpoint path runs a model forward pass, so it
        # wants the GPU-0 pin.
        rc = subprocess.run(cmd_best, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT)).returncode
    print(f"[ms-georgia] best individual bootstrap CI done rc={rc}", flush=True)


def main():
    """CLI entry point: parse args, run the sweep, post-process, report.

    Flow:
      1. Parse '--gpus' (comma list of physical indices; one worker per entry)
         and '--dry-run'.
      2. Print the plan; on '--dry-run' stop after printing (no training).
      3. Enqueue every run spec, spawn one worker thread per '--gpus' entry,
         and join them.
      4. Run 'post_ensemble_and_ci_nb6' to produce the headline ensemble + CIs.
      5. Print a per-depth mean/std/best AUROC table.

    Takes no arguments and returns nothing; everything is driven by 'argv' and
    written to 'SWEEP_DIR'.
    """
    p = argparse.ArgumentParser()
    # Repeats in the list = workers sharing that GPU. Default 0,0,1,1 = two
    # workers each on GPU 0 and GPU 1 (2-per-GPU oversubscription).
    p.add_argument("--gpus",    default="0,0,1,1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    # "0,0,1,1" -> [0, 0, 1, 1]. Length is the worker count; values are the pins.
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    jobs = configs()
    print(f"\n{'=' * 72}")
    print(f"sweep_results_matching_georgia  |  {len(jobs)} jobs  |  GPUs: {args.gpus}")
    print(f"Results: {SWEEP_DIR}")
    # Counter collapses the gpu list into a per-GPU worker tally for the banner.
    counts = Counter(gpu_ids)
    for gid, n in sorted(counts.items()):
        print(f"  GPU{gid}: {n} workers")
    print('=' * 72)

    if args.dry_run:
        # Print-only mode: show what WOULD train, then exit before any subprocess
        # or directory mutation; for sanity-checking the grid/paths.
        for j in jobs:
            print(f"  TRAIN {j['name']:<20}  outdir={j['outdir']}")
        return

    results: list = []
    lock = threading.Lock()
    start_time = time.time()

    # Fill the queue with every job up front; workers drain it concurrently.
    wq: queue.Queue = queue.Queue()
    for j in jobs:
        wq.put(j)
    # One thread per --gpus entry. daemon=True so a Ctrl-C on the main thread is
    # not blocked by lingering workers; name encodes the GPU + worker index for
    # readable tracebacks/logs.
    threads = [threading.Thread(target=worker,
                                 args=(g, wq, results, lock, start_time),
                                 daemon=True, name=f"GPU{g}-w{i}")
               for i, g in enumerate(gpu_ids)]
    for t in threads: t.start()
    for t in threads: t.join()   # block until the queue is fully drained

    # Tally outcomes the same way _write_summary classifies them.
    n_train = sum(1 for r in results if "test_auroc" in r)
    n_fail  = sum(1 for r in results if "test_auroc" not in r and "error" in r)
    eh = (time.time() - start_time) / 3600
    print(f"\n[ms-georgia] training done: {n_train} runs, {n_fail} failures, {eh:.2f}h")

    # Ensemble + bootstrap CIs on the nb=6 set (the new Georgia headline).
    post_ensemble_and_ci_nb6()

    # Per-nb summary.
    # NOTE: statistics is imported here (not at module top) because it is only
    # needed for this end-of-run table.
    import statistics
    # Group test AUROCs by depth label ("nb2"/"nb4"/"nb6") for the table.
    by_group: dict[str, list[float]] = {}
    for r in results:
        if "test_auroc" in r:
            by_group.setdefault(r["group"], []).append(r["test_auroc"])
    print("\nresults_matching on Georgia paper-strict (paper recipe + drop=0.1 + emb=512):")
    for nb in ("nb2", "nb4", "nb6"):
        a = sorted(by_group.get(nb, []))
        if a:
            # pstdev = POPULATION std dev (divides by n, not n-1): we treat the
            # seeds as the full population of runs for this recipe, not a sample.
            m = sum(a)/len(a); sd = statistics.pstdev(a)
            print(f"  {nb}  n={len(a):>2}  mean={m:.4f}  std={sd:.4f}  best={max(a):.4f}  "
                  f"seeds={[f'{x:.4f}' for x in a]}")

    print(f"\n[ms-georgia] ALL results_matching-Georgia work complete  {eh:.2f}h total")


if __name__ == "__main__":
    main()
