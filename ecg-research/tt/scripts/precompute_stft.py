"""
scripts/precompute_stft.py
==========================
One-time builder for the on-disk STFT feature cache used by PTB-XL training.

What this file does
-------------------
An ECG is just voltage over time, but the model is trained on its frequency
content, not the raw trace. For every PTB-XL recording I load the raw 12-lead
waveform, take the per-lead short-time Fourier transform (STFT) magnitude --
how much power sits in each frequency band as the beat unfolds -- and write
that spectrogram as a float32 '.npy' file named '{ecg_id}.npy' in 'cache_dir'.
There are ~21800 records and the STFT is plain CPU arithmetic, so I spread
the work across cores with a 'multiprocessing.Pool'.

Why the cache exists
--------------------
The STFT is fully determined by (signal, fs, nfft, window params), so the same
spectrogram comes out every time -- recomputing it each epoch and each sweep
run is wasted work. 'PTBXLDataset' (src/data/dataset.py) reads
'{cache_dir}/{ecg_id}.npy' straight off disk when it exists and only computes
the STFT on the fly when the cache is missing. Loading a precomputed array is
roughly an order of magnitude faster than recomputing it, which matters most
in the long sweeps where the same records are read thousands of times.

Role in the pipeline / what reads this cache
---------------------------------------------
- Consumer: 'PTBXLDataset.__getitem__' in src/data/dataset.py. It keys the
  cache by 'ecg_id' exactly as I do here ('{ecg_id}.npy'), so the two naming
  schemes MUST stay in lockstep. The dataset can also build the cache lazily,
  one record at a time on first access; this script just front-loads that work
  in parallel so the first training run is not stuck waiting on it.
- Driver: scripts/sweep_nfft_ablation.py (the Table 2 N_FFT ablation of the
  original study) needs a separate cache per N_FFT value and tells the user to
  "pre-build first with scripts/precompute_stft.py --nfft {n}". Each N_FFT
  yields a different '(12, F, T)' shape (F = nfft//2 + 1), so each N_FFT needs
  its own '--cache_dir'; mixing shapes in one directory would silently corrupt
  the run.

How it is run (CLI / harness)
-----------------------------
A standalone CLI script, not imported anywhere. Run it directly from the
'practical_work/' working directory:

    python scripts/precompute_stft.py
    python scripts/precompute_stft.py \\
        --data_dir /path/to/ptb-xl-1.0.3 \\
        --cache_dir /path/to/ptb-xl-stft-cache \\
        --num_workers 64

Re-runs are cheap and safe: any record whose '.npy' already exists is skipped,
so an interrupted build can just be re-launched to finish the rest -- it is
idempotent and resumable.

Note on the sibling implementation
-----------------------------------
src/data/dataset.py also defines a 'compute_stft' helper, but that one takes a
SINGLE lead and is stacked by the caller. The 'compute_stft' in THIS file loops
over all 12 leads internally and returns the stacked '(12, F, T)' array. The
numerical recipe (window, overlap, nfft, magnitude, float32) is identical, so
the two paths produce byte-compatible caches; only the loop placement differs.
"""

import os
import ast            # NOTE: imported but unused here; the dataset module uses
                      # ast.literal_eval to parse scp_codes. Kept only to mirror
                      # the dataset's import block. NOTE: drop it if matching
                      # dataset.py stops being a goal.
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count   # CPU fan-out; STFT is pure-CPU SciPy work
from functools import partial                 # to pin the per-record constants onto the worker
import wfdb                                    # PhysioNet WFDB reader for the raw ECG records
from scipy.signal import stft as scipy_stft    # the actual STFT implementation we cache
from tqdm import tqdm                          # progress bar over the ~21.8k records
from omegaconf import OmegaConf                # to read the default data paths from config.yaml

# Default data paths come from config.yaml (set data.root there) so there is one
# place to point at the data; the argparse defaults below read from here.
_CFG = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def compute_stft(signal: np.ndarray, fs: int = 100,
                 nfft: int = 480) -> np.ndarray:
    """Compute the magnitude STFT for all 12 leads of one recording.

    Args:
        signal: raw waveform of shape (T_samples, 12), i.e. time x leads.
            For 100 Hz PTB-XL records T_samples = 1000 (a 10 s strip).
        fs: sampling rate in Hz. Only passed through to SciPy so the returned
            frequency axis is labeled correctly; it does not change the
            magnitudes I keep.
        nfft: FFT length applied to each (zero-padded) 64-sample window. Sets
            the frequency resolution: F = nfft // 2 + 1 bins. The original
            study (Section 4.3) uses nfft=480 -> F=241; the Table 2 ablation
            sweeps {240, 360, 480, 512}.

    Returns:
        float32 array of shape (12, F, 59):
            12 = leads, F = nfft//2 + 1 frequency bins, 59 = time frames.
        The time-frame count 59 is fixed by nperseg/noverlap and the 1000-
        sample length, NOT by nfft, so it stays 59 across every N_FFT value.

    Window recipe (Section 4.3 of the original study; must match
    src/data/dataset.py):
        nperseg=64    -> 64-sample analysis window
        noverlap=48   -> hop = 64 - 48 = 16 samples (75% overlap)
        boundary=None -> do not pad/extend the signal edges
        padded=False  -> do not zero-pad the signal to fit a whole window
        I take np.abs(Zxx): the magnitude spectrogram only, phase discarded.
        Keeping magnitude alone is the deliberate choice -- the model is trained
        on how much energy sits in each band over time, not on phase.
    """
    # Build one (F, 59) magnitude spectrogram per lead, then stack the 12.
    # NOTE: this loops over leads inside the function; the sibling helper in
    # src/data/dataset.py instead takes a single lead and is stacked by the
    # caller. Both yield the same per-lead array, so teh caches are compatible.
    leads = []
    for lead in range(signal.shape[1]):           # signal.shape[1] == 12 leads
        # scipy_stft returns (freqs, times, Zxx); I only need Zxx, the complex
        # STFT of this single lead with shape (F, 59).
        _, _, Zxx = scipy_stft(
            signal[:, lead], fs=fs, nperseg=64, noverlap=48,
            nfft=nfft, boundary=None, padded=False
        )
        # np.abs -> magnitude (drops phase); cast to float32 to halve the
        # on-disk cache size versus float64, with no accuracy cost downstream.
        leads.append(np.abs(Zxx).astype(np.float32))
    # Stack along a new leading axis: 12 x (F, 59) -> (12, F, 59).
    return np.stack(leads, axis=0)  # (12, F, 59)


def process_one(args, data_dir: str, cache_dir: str, fs: int, nfft: int):
    """Build and persist the STFT cache for a single recording (worker body).

    This is what each pool worker runs. It is self-contained and returns a
    result tuple instead of raising, so one unreadable record cannot bring
    down the whole pool; the parent collects the failures and reports them.

    Args:
        args: a (ecg_id, filename) pair. Packed as one positional argument
            because 'imap_unordered' passes a single item per call; the other
            parameters are bound up front via functools.partial in main().
        data_dir: PTB-XL root directory holding the WFDB records.
        cache_dir: output directory for the '{ecg_id}.npy' cache file.
        fs: expected sampling rate (Hz); asserted against the record header.
        nfft: FFT length forwarded to compute_stft (sets F = nfft//2 + 1).

    Returns:
        (ecg_id, success, error):
            success=True, error=None      on a cache hit or a fresh write.
            success=False, error=<str>    if anything went wrong (the message
                                          is the stringified exception).
    """
    ecg_id, filename = args                                   # unpack the (id, path) pair
    cache_path = os.path.join(cache_dir, f"{ecg_id}.npy")     # flat per-ecg_id cache file

    # Skip if already cached. This is what makes the script resumable: an
    # interrupted run can be re-launched and only the missing records get
    # recomputed. NOTE: I do not check the existing file's shape here, so a
    # cache built under a DIFFERENT nfft would be wrongly trusted as valid;
    # the contract is "one cache_dir per nfft" (see module docstring).
    if os.path.exists(cache_path):
        return ecg_id, True, None

    try:
        record_path = os.path.join(data_dir, filename)
        # wfdb.rdsamp returns (signals, fields); signals is (n_samples, 12).
        signals, fields = wfdb.rdsamp(record_path)
        # Guard against silently caching a wrong-rate record (e.g. a 500 Hz
        # file when I asked for 100 Hz), which would give the wrong T frames.
        assert fields["fs"] == fs, f"Expected fs={fs}, got {fields['fs']}"
        signals = signals.astype(np.float32)                 # match the cached dtype
        stft = compute_stft(signals, fs, nfft)               # (12, F, 59)
        # NOTE: writes straight to cache_path (no temp-file + atomic rename).
        # Each ecg_id is owned by exactly one worker, so nothing else writes
        # the same file here. The dataset's lazy-build path DOES use the
        # tmp+os.replace dance, because there many DataLoader workers can race
        # on the same record.
        # NOTE: adopt the same tmp-then-rename write here too, so a SIGKILL
        # mid-write cannot leave a truncated .npy that the skip check above
        # would later trust.
        np.save(cache_path, stft)
        return ecg_id, True, None
    except Exception as e:
        # Report rather than propagate: one unreadable record should not abort
        # the remaining ~21.8k. The parent collects these and prints them.
        return ecg_id, False, str(e)


def main():
    """CLI entry point: parse args, enumerate records, fan out, verify.

    Steps:
      1. Parse CLI options (paths, sampling rate, nfft, worker count).
      2. Read 'ptbxl_database.csv' to get the (ecg_id, filename) pairs.
      3. Skip early if everything is already cached.
      4. Run 'process_one' over the pairs in a process pool with a progress bar.
      5. Report failures and sanity-check the first cached file's shape/dtype.

    No meaningful return value; the side effect is the populated 'cache_dir'.
    """
    parser = argparse.ArgumentParser()
    # Defaults come from config.yaml (set data.root there). The Table 2 N_FFT
    # sweep overrides --cache_dir per nfft (each shape needs its own dir).
    parser.add_argument("--data_dir", default=str(_CFG.data.data_dir))
    parser.add_argument("--cache_dir", default=str(_CFG.data.cache_dir))
    # fs=100 selects the low-rate (filename_lr) records; fs=500 the high-rate.
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--nfft", type=int, default=480,
                        help="FFT length; F = nfft//2 + 1 freq bins per "
                             "lead. Use a separate cache_dir for each nfft.")
    # Cap at 192 workers OR the machine's core count, whichever is smaller, so
    # this does not oversubscribe a smaller box. The dyopto server has many
    # cores, so in practice the 192 cap is what binds.
    parser.add_argument("--num_workers", type=int, default=min(192, cpu_count()))
    args = parser.parse_args()

    # Create the cache dir up front. exist_ok=True so re-runs do not error.
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from {args.data_dir}...")
    df = pd.read_csv(os.path.join(args.data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    # Pick the column holding the relative record path for the chosen rate:
    # filename_hr = 500 Hz records, filename_lr = 100 Hz records (the default).
    filename_col = "filename_hr" if args.fs == 500 else "filename_lr"
    # One work item per recording: (ecg_id, relative_path). ecg_id is the cache key.
    pairs = [(ecg_id, row[filename_col]) for ecg_id, row in df.iterrows()]

    print(f"Total recordings: {len(pairs)}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Using {args.num_workers} CPU workers")

    # Count what is already cached up front. This lets me short-circuit a
    # fully built cache (common when re-running the N_FFT=480 default that
    # ships pre-built) and print a meaningful "X/N already cached" line.
    already_cached = sum(
        1 for ecg_id, _ in pairs
        if os.path.exists(os.path.join(args.cache_dir, f"{ecg_id}.npy"))
    )
    print(f"Already cached: {already_cached}/{len(pairs)}")

    # Fast exit: nothing to do. The pool/verify steps below also assume at
    # least the first record exists, so returning here doubles as a guard.
    if already_cached == len(pairs):
        print("All recordings already cached.")
        return

    # Process in parallel.
    # Bind the constants (data_dir, cache_dir, fs, nfft) onto process_one so
    # the pool only has to pass the per-record (ecg_id, filename) pair. partial
    # also keeps the worker picklable, which Pool needs to ship it to the
    # child processes.
    worker_fn = partial(process_one, data_dir=args.data_dir,
                        cache_dir=args.cache_dir, fs=args.fs,
                        nfft=args.nfft)

    failed = []
    with Pool(processes=args.num_workers) as pool:
        # imap_unordered: results come back as soon as any worker finishes,
        # which keeps the tqdm bar moving smoothly and avoids holding all
        # results in memory. Order does not matter since each worker writes
        # its own file and I only collect failures.
        # chunksize=50: hand 50 records to a worker at a time to amortize the
        # per-task IPC overhead; the STFT is short enough that a chunksize of 1
        # would make scheduling, not computation, the bottleneck.
        for ecg_id, success, error in tqdm(
            pool.imap_unordered(worker_fn, pairs, chunksize=50),
            total=len(pairs),
            desc="Precomputing STFT"
        ):
            if not success:
                failed.append((ecg_id, error))   # accumulate for the report below

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        # Print only the first 20 failures to keep the log readable; the count
        # above is the source of truth for how many failed in total.
        print("Failed recordings:")
        for ecg_id, error in failed[:20]:
            print(f"  {ecg_id}: {error}")

    # Verify the shape of the first cached file -- a cheap end-to-end sanity
    # check. If the cache was built under the wrong nfft, the F dimension
    # printed here will not match the expected (12, F, 59), surfacing the
    # mistake right away.
    first_ecg_id = pairs[0][0]
    sample = np.load(os.path.join(args.cache_dir, f"{first_ecg_id}.npy"))
    expected_F = args.nfft // 2 + 1                # same F = nfft//2 + 1 contract
    print(f"\nSample shape: {sample.shape} (expected: (12, {expected_F}, 59))")
    print(f"Sample dtype: {sample.dtype}")
    # Total on-disk footprint of the cache, in GB (1e9 bytes). NOTE: this sums
    # every file in cache_dir, so it would also count any stray non-.npy files;
    # in practice the directory holds only the per-record arrays.
    print(f"Cache size: {sum(os.path.getsize(os.path.join(args.cache_dir, f)) for f in os.listdir(args.cache_dir)) / 1e9:.2f} GB")


if __name__ == "__main__":
    # Standard script guard. The 'multiprocessing.Pool' started inside main()
    # relies on this on platforms that spawn child processes (so the children
    # re-import the module without re-running main()); on Linux 'fork' it is
    # not strictly required but is kept for portability.
    main()