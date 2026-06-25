"""
scripts/sweep_nfft_ablation.py
============================
Reproduce Table 2 of the original study (the N_FFT hyperparameter search),
but under the corrected 'label_aggregation = lik_eq_100' labels rather than
the original study's own label handling.

Clinically, N_FFT sets how finely the short-time Fourier transform resolves
the frequency content of each ECG lead -- a larger window buys frequency
detail at the cost of time detail. Section 4.4 of the original study sweeps
N_FFT over {240, 360, 480, 512} under its hyperparameter-search protocol (the
original recipe plus early_stopping_patience=2, a flat 2e-4 base LR, and the
study's StepLR(step_size=2, gamma=0.8)) and reports that 480 wins by about
0.005 AUROC. Here I re-run that sweep on my own corrected labels to see
whether the same N_FFT comes out on top.

The sweep itself:
- 4 N_FFT values x 3 seeds = 12 runs.
- Each N_FFT needs its own STFT cache: a cached array has shape (12, F, T)
  with F = nfft//2 + 1, so the bytes for one N_FFT cannot be read as another.
  nfft=480 reuses the existing default cache; the other three caches
  live under data_cache/ in the repo root.
- I never set model.input_size by hand. train.py derives it from
  cfg.data.nfft, so the sweep stays a clean one-axis ablation.


Usage
-----
    python scripts/sweep_nfft_ablation.py --gpus 0,0,0,1,1,1
    python scripts/sweep_nfft_ablation.py --gpus 0,1 --dry-run
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

PROJECT   = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT / "results" / "sweep_nfft_ablation"
PYTHON    = sys.executable

PAPER_PTBXL = {
    "training.optimizer":               "adam",
    "training.learning_rate":           2e-4,
    "training.lr_scheduler":            "step",     # StepLR; gamma below keeps it close to flat
    "training.step_lr_step_size":       2,          # decay every 2 epochs ...
    "training.step_lr_gamma":           0.8,        # ... by x0.8 (Section 4.4 schedule of the original study)
    "training.weight_decay":            0.0,
    "training.normalization":           "zero_mean_unit_var",  # z-score per batch (see train.py)
    "training.loss_fn":                 "bce",      # multi-label: BCE, not softmax CE
    "training.early_stopping_patience": 2,          # the original study's hyper-search protocol (short, for speed)
    "training.num_epochs":              500,        # ceiling only; early stop ends runs far sooner
    "model.embedding_dim":              256,        # not stated by the original study; config.yaml default
    "model.num_heads":                  4,
    "model.num_blocks":                 2,
    "model.fusion_type":                "layer",    # eq. 13 of the original study (iterative sLSTM/mLSTM fusion)
    "model.dropout":                    0.5,
    "model.mask_ratio":                 0.2,        # input masking: fraction of tokens masked ...
    "model.mask_prob":                  0.8,        # ... applied with this probability per batch
    "model.pooling":                    "mean",
    "data.label_aggregation":           "lik_eq_100",  # corrected labels: keep diagnoses w/ likelihood==100
    # NOTE: model.input_size is deliberately absent. train.py derives it from
    # data.nfft (input_size = 12 * (nfft//2 + 1)); hard-coding it here would
    # break the 240/360/512 jobs whose F differs from the 480 default.
}

# Per-N_FFT STFT cache directory. One cache per N_FFT, because a cached array
# has shape (12, F, T) with F = nfft//2 + 1, so the bytes for one N_FFT are
# not interpretable as another. The 480 cache is the project's default PTB-XL
# cache, read from config.yaml; the other three are repo-local under
# data_cache/.
_DEFAULT_CACHE = str(OmegaConf.load(PROJECT / "config" / "config.yaml").data.cache_dir)
NFFT_TO_CACHE = {
    240: str(PROJECT / "data_cache" / "ptb-xl-stft-cache-nfft240"),   # F = 121
    360: str(PROJECT / "data_cache" / "ptb-xl-stft-cache-nfft360"),   # F = 181
    480: _DEFAULT_CACHE,                             # default of the original study; from config.yaml (F = 241)
    512: str(PROJECT / "data_cache" / "ptb-xl-stft-cache-nfft512"),   # F = 257
}

# The two sweep axes. Their Cartesian product (4 x 3 = 12) is the job list.
# Three fixed seeds let me report a per-N_FFT mean +/- std at the end rather
# than a single noisy point estimate -- one run could win on luck of the seed.
NFFT_VALUES = (240, 360, 480, 512)
SEEDS = (42, 123, 456)


def configs() -> list[dict]:
    """Build the full job list: every (N_FFT, seed) combination.

    Returns
    -------
    list[dict]
        One spec per run. Each spec has:
        - 'name'      : unique per-run id, also the run's output subfolder
                          name ('nfft_ablation{n}_s{s}').
        - 'group'     : the N_FFT bucket a run belongs to
                          ('nfft_ablation{n}'), used to aggregate seeds.
        - 'overrides' : the PAPER_PTBXL recipe with this run's varying keys
                          (data.nfft / data.cache_dir / training.seed) layered
                          on top. This dict is handed verbatim to
                          run_training.py as 'key=value' overrides.

    Order is N_FFT-major, seed-minor; nothing downstream relies on the order
    (work is pulled off a queue), it only affects the dry-run print order.
    """
    out = []
    for n in NFFT_VALUES:
        for s in SEEDS:
            out.append({
                "name":  f"nfft_ablation{n}_s{s}",
                "group": f"nfft_ablation{n}",
                "overrides": {
                    # PAPER_PTBXL first, then the per-run keys, so the varying
                    # keys win if they ever collide with the recipe.
                    **PAPER_PTBXL,
                    "data.nfft":      n,                # the swept axis
                    "data.cache_dir": NFFT_TO_CACHE[n],  # matching cache for this nfft
                    "training.seed":  s,                # the repeat axis
                },
            })
    return out


def worker(gpu_id, work_queue, results, lock, start_time):
    """One scheduling thread bound to a single GPU.

    Parameters
    ----------
    gpu_id      : int   GPU index this worker pins its child processes to.
    work_queue  : queue.Queue  shared FIFO of job specs (thread-safe).
    results     : list  shared results accumulator (guard writes with 'lock').
    lock        : threading.Lock  protects 'results' and the summary write.
    start_time  : float  wall-clock t0 (from time.time()) for elapsed-hour logs.
    """
    while True:
        try:
            # Non-blocking pop: the queue is fully populated before any worker
            # starts, so "empty" unambiguously means "all jobs claimed" -- no
            # producer is still adding, so I can stop rather than block.
            spec = work_queue.get_nowait()
        except queue.Empty:
            break
        _run_training(gpu_id, spec, results, lock, start_time)
        work_queue.task_done()


def _run_training(gpu_id, spec, results, lock, start_time):
    """Execute one training run as a child process and record its result."""
    name, group, overrides = spec["name"], spec["group"], spec["overrides"]
    run_dir     = SWEEP_DIR / name
    result_file = run_dir / "result.json"   # train.py writes this iff I pass training.result_file
    log_file    = run_dir / "train.log"     # combined stdout+stderr of the child
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume path: a completed run is reloaded, not recomputed. -----------
    if result_file.exists():
        eh = (time.time() - start_time) / 3600
        print(f"[GPU{gpu_id}] SKIP  {name:<30} +{eh:.2f}h (done)", flush=True)
        with open(result_file) as f:
            result = json.load(f)
        # Re-stamp identity fields so a reloaded result carries the same
        # name/group/overrides as a freshly produced one (train.py's JSON has
        # its own config snapshot but not these harness-level labels).
        # Without this, a resumed run would land in the summary unlabeled.
        result.update({"name": name, "group": group, "overrides": overrides})
        with lock:
            results.append(result)
        return

    # --- Build the child command. -------------------------------------------
    # These three keys steer the child's per-run artifacts into THIS run's
    # folder; in particular training.result_file is the contract that makes
    # train.py emit the metrics JSON I read back.
    extra = [
        f"training.checkpoint_dir={run_dir / 'checkpoints'}",
        f"training.log_dir={run_dir / 'logs'}",
        f"training.result_file={result_file}",
    ]
    # 'python -m scripts.run_training k1=v1 k2=v2 ... <extra>' -- overrides are
    # OmegaConf dot-list assignments merged onto config.yaml by run_training.py.
    cmd = [PYTHON, "-m", "scripts.run_training"] \
          + [f"{k}={v}" for k, v in overrides.items()] + extra
    # Copy the parent environment and pin the child to one physical GPU. The
    # child then sees that one device as cuda:0 regardless of gpu_id, which is
    # what lets several workers target different cards without coordination.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    eh = (time.time() - start_time) / 3600
    print(f"[GPU{gpu_id}] START {name:<30} +{eh:.2f}h", flush=True)
    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            # Header lines make a stray log self-describing (exact command, GPU,
            # start time) without needing the harness's own stdout to interpret it.
            lf.write(f"CMD: {' '.join(str(c) for c in cmd)}\n")
            lf.write(f"GPU: {gpu_id}\nSTARTED: {datetime.now().isoformat()}\n\n")
            lf.flush()

            # Redirect both child streams into the same log file. cwd=PROJECT so
            # the '-m scripts.run_training' import path and the relative
            # config/config.yaml default both resolve.
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf,
                                    cwd=str(PROJECT))

            # 10800 s = 3 h hard wall per run. With patience=2 a real run is far
            # shorter; the cap exists to reclaim a wedged GPU rather than to bound
            # a healthy run.
            proc.wait(timeout=10800)

        elapsed = time.time() - t0
        if result_file.exists():
            # Good case: the child wrote its metrics JSON.
            with open(result_file) as f:
                result = json.load(f)
            result.update({"name": name, "group": group, "overrides": overrides})

            # NaN sentinel so the format string below never blows up if a
            # malformed JSON is missing test_auroc.
            au = result.get("test_auroc", float("nan"))
            print(f"[GPU{gpu_id}] DONE  {name:<30} AUROC={au:.4f}  {elapsed/60:.1f}min",
                  flush=True)
        else:
            # Child exited (no timeout) but produced no result.json -> it failed
            # internally. Capture the return code for later triage.
            result = {"name": name, "group": group, "overrides": overrides,
                      "error": "no result.json", "rc": proc.returncode}
            print(f"[GPU{gpu_id}] FAIL  {name:<30} rc={proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        # Hit the 3 h wall. Kill the child and reap it so the GPU is released;
        # without the kill+wait the process would linger and keep holding VRAM.
        proc.kill(); proc.wait()
        result = {"name": name, "group": group, "overrides": overrides, "error": "timeout"}
        print(f"[GPU{gpu_id}] TIMEOUT {name}", flush=True)
    except Exception as e:
        # Anything else (e.g. failure to spawn). Record it and keep the sweep
        # alive.
        result = {"name": name, "group": group, "overrides": overrides, "error": str(e)}
        print(f"[GPU{gpu_id}] ERROR {name}: {e}", flush=True)

    # Append the result and rewrite the rolling summary so a crash mid-sweep
    # still leaves an up-to-date summary.json on disk.
    # The lock serializes both the list mutation and the file write.
    with lock:
        results.append(result)
        _write_summary(results)


def _write_summary(results):
    """Rewrite 'summary.json' from the results so far.

    Called under the shared lock after every run completes, so the file on
    disk always reflects the latest state and a sweep killed partway through
    still leaves a usable summary. Completed runs are sorted best-AUROC-first;
    failed runs (no 'test_auroc' but an 'error') are listed separately.

    'results' is the shared accumulator list; this function only reads it.
    """
    # A run "completed" iff it has a test_auroc; rank descending so the top of
    # the file is the best run. NOTE: the per-nfft mean/std is computed at the
    # end in main(), not here -- this file is the raw per-run record.
    done = sorted([r for r in results if "test_auroc" in r],
                  key=lambda r: r.get("test_auroc", 0), reverse=True)
    # Mirror image: has an error and no metric. The two predicates are mutually
    # exclusive by construction (a result dict gets one or the other).
    fail = [r for r in results if "test_auroc" not in r and "error" in r]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump({
            "updated":   datetime.now().isoformat(),
            "training":  {"completed": len(done), "failed": len(fail), "runs": done},
            "all_failed": fail,
        }, f, indent=2)
    # NOTE: this overwrites the file each call rather than appending; for a
    # 12-run sweep the rewrite cost is negligible and it keeps the on-disk JSON
    # always valid (no partial-append corruption window).


def main():
    """CLI entry point: parse args, validate caches, run the sweep, report.

    Flow:
    1. Parse '--gpus' (worker list) and '--dry-run'.
    2. Build the 12-job list and, if '--dry-run', print it and exit.
    3. Verify every required STFT cache exists (hard error if missing) and is
       populated (soft warning if suspiciously small).
    4. Fan the jobs across one worker thread per '--gpus' entry; block until
       all finish.
    5. Print a per-N_FFT mean / std / best table -- the read-back I compare
       against Table 2 of the original study.
    """
    p = argparse.ArgumentParser(description="Table 2 N_FFT sweep")
    p.add_argument("--gpus",    default="0,0,0,1,1,1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    # "0,0,0,1,1,1" -> [0,0,0,1,1,1]: one worker thread per element, so a
    # repeated id oversubscribes that GPU (three trainings per H100).
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    jobs = configs()
    print(f"\n{'=' * 72}")
    print(f"sweep_nfft_ablation  |  {len(jobs)} jobs  |  GPUs: {args.gpus}")
    print(f"Results: {SWEEP_DIR}")
    print('=' * 72)

    if args.dry_run:
        # Show only the keys that actually vary per job (the data.* axes and the
        # seed); the constant PAPER_PTBXL recipe would just be clutter here.
        for j in jobs:
            d = {k: v for k, v in j["overrides"].items() if k.startswith("data.") or k == "training.seed"}
            print(f"  TRAIN {j['name']:<30}  {d}")
        return

    # Sanity-check that every required cache dir exists and has files.
    # A missing cache is fatal (the run would crash on first access); a small
    # cache is only a warning because a partially built cache may still be
    # filling -- but I name the exact precompute command to finish it.
    for n, cd in NFFT_TO_CACHE.items():
        if not Path(cd).is_dir():
            print(f"ERROR: cache dir missing for nfft={n}: {cd}"); return
        n_files = len(list(Path(cd).iterdir()))
        # 100 is a loose floor: PTB-XL has ~21.8k records, so a real cache holds
        # tens of thousands of .npy files; under 100 means it is essentially empty.
        if n_files < 100:
            print(f"WARNING: cache dir for nfft={n} only has {n_files} files; "
                  f"pre-build first with scripts/precompute_stft.py --nfft {n}.")
            # NOTE: I warn but continue -- the dataset can also build STFTs
            # lazily on first access, just far more slowly than a warm cache.

    results: list = []                 # shared accumulator (guarded by 'lock')
    lock = threading.Lock()            # serializes results + summary.json writes
    start_time = time.time()           # t0 for the "+{h}h" progress timestamps

    # Informational count of how many runs still need doing (the resume logic
    # lives in _run_training; this is just a heads-up line).
    pending = [j for j in jobs if not (SWEEP_DIR / j["name"] / "result.json").exists()]
    print(f"\n{len(pending)}/{len(jobs)} to run ({len(jobs) - len(pending)} done).",
          flush=True)

    # Load ALL jobs (not just 'pending') into the queue; already-done jobs are
    # cheaply short-circuited by the resume check in _run_training, which also
    # folds their existing results back into the summary.
    wq = queue.Queue()
    for j in jobs:
        wq.put(j)

    # One daemon thread per --gpus entry. daemon=True so a Ctrl-C on the main
    # thread does not hang on stragglers; name is for readable tracebacks.
    threads = [threading.Thread(target=worker,
                                 args=(g, wq, results, lock, start_time),
                                 daemon=True, name=f"GPU{g}-w{i}")
               for i, g in enumerate(gpu_ids)]
    for t in threads: t.start()
    for t in threads: t.join()   # block until every job is drained from the queue

    eh = (time.time() - start_time) / 3600
    n_train = sum(1 for r in results if "test_auroc" in r)
    n_fail  = sum(1 for r in results if "test_auroc" not in r and "error" in r)
    print(f"\nDONE  {eh:.2f}h  |  {n_train} runs  |  {n_fail} failures")

    # Print per-N_FFT mean +/- std for a quick read-back.
    # Local imports: these are only needed for the final aggregation, so they
    # stay out of the module-level import block.
    import statistics, re
    by_nfft: dict[int, list[float]] = {}
    for r in results:
        if "test_auroc" not in r:
            continue   # skip failed runs; they contribute no AUROC
        # Recover the N_FFT from the run name (e.g. "nfft_ablation480_s42") rather
        # than trusting overrides -- the name is the single source of identity
        # written to disk. The regex anchors on the "_s<digits>" suffix so it
        # cannot accidentally match a seed value.
        m = re.search(r"nfft(\d+)_s\d+", r["name"])
        if m:
            by_nfft.setdefault(int(m.group(1)), []).append(r["test_auroc"])
    print("\nPaper Table 2 reproduction (lik_eq_100, patience=2):")
    print(f"  {'N_FFT':>6}  {'mean':>7}  {'std':>7}  {'best':>7}  seeds")
    for nfft in sorted(by_nfft):
        a = sorted(by_nfft[nfft])
        # pstdev = population std (divide by N, not N-1). With only 3 seeds the
        # choice barely matters and population std avoids the N=1 edge case
        # where sample std is undefined.
        # NOTE: report a bootstrap 95% CI alongside mean/std here so the table
        # matches the CI convention used in bootstrap_ci.py.
        print(f"  {nfft:>6}  {sum(a)/len(a):.4f}  {statistics.pstdev(a):.4f}  "
              f"{max(a):.4f}  {[f'{x:.4f}' for x in a]}")


if __name__ == "__main__":
    main()
