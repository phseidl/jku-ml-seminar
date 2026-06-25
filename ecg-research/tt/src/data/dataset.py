"""
src/data/dataset.py
===================
PTB-XL data loader for the xLSTM-ECG reproduction. This is where a clinical
12-lead ECG and its cardiologist annotations become the (input, label) pairs
the network trains on.

Usage
-----
    from src.data.dataset import PTBXLDataset
    ds = PTBXLDataset(
        data_dir="/path/to/ptb-xl-1.0.3",
        split="train",
        cache_dir="/path/to/ptb-xl-stft-cache",     # required for fast loads
        label_aggregation="lik_eq_100",              # project default
        nfft=480,                                    # paper default
    )
"""

import os
import ast                       # literal_eval the scp_codes dict column (see below)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb                       # PhysioNet WaveForm DataBase reader for the .dat/.hea record pairs
from scipy.signal import stft as scipy_stft  # aliased so it does not clash with the compute_stft wrapper


# The 5 PTB-XL superclasses, in a fixed order that doubles as the label-vector
# layout. his lis is the single source of truth for column ordering: every
# (5,) label, the per-class AUROC table, and the confusion-matrix axes are read
# off it, so reordering it would silently invalidate every cached label and
# every reported number. The order also breaks likelihood ties in
# build_label_vector(primary_max_lik): NORM (index 0) beats MI, and so on.
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# PTB-XL ships a 10-way stratified fold column (strat_fold, 1..10) built so each
# fold preserves the class mix. The standard protocol trains on folds 1-8, uses
# fold 9 for validation/model selection, and holds out fold 10 as the test set.
# I follow it exactly so my test fold (N=1711 records under lik_eq_100) is
# comparable with the published literature.
FOLD_TRAIN = list(range(1, 9))
FOLD_VAL = [9]
FOLD_TEST = [10]

# Records dropped up front. ecg_id 10581 had a malformed signal line in its .hea
# header that older wfdb could not parse ("invalid syntax in signal line"). The
# pinned wfdb (4.3.1) now reads it fine, but I keep the exclusion so the cohort
# stays identical to the one every result was produced on. It lives in train
# fold 1 and carries no likelihood-100 superclass code, so under the default
# lik_eq_100 rule it is label-less and dropped regardless -- this exclusion moves
# neither the test fold (N=1711) nor any headline number; it only removes one
# record under the historical lik_gt_0 rule. To regenerate the list for a new
# release, try wfdb.rdsamp on every record.
# NOTE: a small scripts/scan_broken_records.py that walks every filename_lr,
#       tries wfdb.rdsamp, and prints the failing ecg_ids would let this set be
#       regenerated mechanically (instead of by hand) when the PTB-XL version
#       (currently 1.0.3) bumps, not reallz likely in our case.
KNOWN_BROKEN_ECG_IDS = {10581}


def load_ptbxl_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load ptbxl_database.csv (the master record table) and parse scp_codes.

    Args:
        data_dir: directory holding the unpacked PTB-XL release; must contain
                  ptbxl_database.csv at its top level.

    Returns:
        A DataFrame indexed by integer ecg_id, one row per recording, with the
        usual PTB-XL columns (scp_codes, strat_fold, filename_lr, filename_hr,
        patient demographics, ...). The scp_codes column is converted in place
        from its on-disk string form into a real dict.

    Called by PTBXLDataset.__init__.
    """
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    # index_col="ecg_id" keys the DataFrame by the same integer id used to name
    # the STFT cache files ({ecg_id}.npy), so df.index aligns 1:1 with the cache.
    df = pd.read_csv(csv_path, index_col="ecg_id")
    # scp_codes is stored in the CSV as the *string* repr of a Python dict, e.g.
    # "{'NORM': 100.0}" or "{'IMI': 80.0, 'LBBB': 100.0}". literal_eval turns
    # that text back into a real dict {code: likelihood}. I use ast.literal_eval
    # rather than eval() because it only parses literals -- it cannot execute
    # arbitrary code that happens to sit in the CSV.
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(data_dir: str) -> pd.DataFrame:
    """
    Load scp_statements.csv, the SCP-code dictionary that maps each individual
    SCP-ECG statement code to its diagnostic superclass.

    Args:
        data_dir: directory containing scp_statements.csv.

    Returns:
        A DataFrame indexed by SCP code (the unnamed first column). The only
        column consumed downstream is diagnostic_class -- one of NORM/MI/STTC/
        CD/HYP, or NaN for non-diagnostic codes such as rhythm or form
        statements. PTBXLDataset.__init__ reduces this to a plain
        {scp_code: superclass} dict.

    Called by PTBXLDataset.__init__.
    """
    path = os.path.join(data_dir, "scp_statements.csv")
    # index_col=0 -> use the leftmost (unnamed) column, which holds the SCP code.
    return pd.read_csv(path, index_col=0)


def build_label_vector(scp_codes: dict, scp_to_super: dict,
                       aggregation: str = "lik_gt_0") -> np.ndarray:
    """
    Turn one recording's scp_codes dict {code: likelihood} into a (5,) label.

    This is where the diagnosis label is actually decided. A cardiologist rarely
    commits to one clean class -- PTB-XL annotations come hedged with a
    likelihood from 0 to 100 -- so this step converts that clinical hedging into
    the hard 0/1 targets a classifier can be scored against. Because every
    PTB-XL number I report depends on the rule chosen here, I keep all three
    rules explicit rather than bury one as a default (tests/test_data.py pins
    this contract).

    Args:
        scp_codes:    one record's annotations as {SCP_code: likelihood}, the
                      likelihood being a percent in [0, 100] (e.g.
                      {"NORM": 100.0}). This is the literal-eval'd scp_codes cell
                      from ptbxl_database.csv.
        scp_to_super: mapping {SCP_code: superclass_name}, i.e. the
                      diagnostic_class column of scp_statements.csv reduced to a
                      dict. Codes absent from it are non-diagnostic (rhythm/form)
                      and ignored.
        aggregation:  one of:
            "lik_gt_0"        : multi-label; any code with likelihood strictly
                                > 0 counts (a code listed at likelihood 0 is
                                dropped). Close to, but not identical with, the
                                Strodthoff PTB-XL benchmark, which aggregates by
                                code presence and would keep a likelihood-0 code
                                this rule drops; on the 5 superclasses, where
                                likelihood-0 codes are rare, the two nearly
                                coincide. Also this function's parameter default,
                                for backwards compatibility.
            "lik_eq_100"      : multi-label; only codes the annotator was fully
                                sure of (likelihood == 100) count. Drops the
                                uncertain reads, so each class has fewer
                                positives than under lik_gt_0, HYP most of all.
                                The project default; train.py selects it via
                                config.yaml.
            "primary_max_lik" : single-label; give the record the superclass of
                                its highest-likelihood code (ties broken by
                                SUPERCLASSES order) -- one diagnosis per record.

    Returns:
        A (5,) float32 ndarray of 0.0/1.0 entries in SUPERCLASSES order. All-zero
        when no code maps to a superclass under the chosen rule -- those records
        have no usable label and PTBXLDataset.__init__ drops them.
    """
    # Start everyone negative across all 5 superclasses; the branches below flip
    # individual positions to 1.0. float32 (not bool/int) because the loss
    # (nn.BCELoss on the model's sigmoid outputs) expects float targets.
    label = np.zeros(len(SUPERCLASSES), dtype=np.float32)

    if aggregation == "lik_gt_0":
        # Multi-label union: a record can be positive for several superclasses.
        for code, lik in scp_codes.items():
            # Guard order matters: 'code in scp_to_super' drops the
            # non-diagnostic (rhythm/form) codes before indexing the dict.
            if lik > 0 and code in scp_to_super:
                super_cls = scp_to_super[code]
                if super_cls in SUPERCLASSES:                 # ignore any class outside the 5
                    label[SUPERCLASSES.index(super_cls)] = 1.0  # set, idempotent on repeats
        return label

    if aggregation == "lik_eq_100":
        # Same as lik_gt_0 but stricter: only fully-confident annotations
        # (likelihood == 100) count. Exact equality to 100 is deliberate --
        # PTB-XL likelihoods are discrete (0/15/35/50/80/100...), so '== 100' is
        # well-defined and not a float-comparison hazard here.
        for code, lik in scp_codes.items():
            if lik == 100 and code in scp_to_super:
                super_cls = scp_to_super[code]
                if super_cls in SUPERCLASSES:
                    label[SUPERCLASSES.index(super_cls)] = 1.0
        return label

    if aggregation == "primary_max_lik":
        # Single-label: collapse to exactly one superclass per record. The
        # highest-likelihood SCP code wins, ties broken by SUPERCLASSES order.
        # Build (likelihood, superclass) candidates, dropping non-diagnostic and
        # out-of-set codes the same way the multi-label branches do.
        cands = [(lik, scp_to_super[c])
                 for c, lik in scp_codes.items()
                 if c in scp_to_super and scp_to_super[c] in SUPERCLASSES]
        if not cands:
            return label   # no diagnostic code -> all-zero, record gets dropped upstream
        # Sort key (-lik, SUPERCLASSES.index): primary key is descending
        # likelihood (negated so a plain ascending sort puts the largest first);
        # the secondary key makes ties deterministic by preferring the
        # earlier-listed superclass (NORM before MI before ...).
        cands.sort(key=lambda x: (-x[0], SUPERCLASSES.index(x[1])))
        label[SUPERCLASSES.index(cands[0][1])] = 1.0          # winner only
        return label

    # Defensive: an unrecognized aggregation string is a configuration error,
    # not a silent no-op -- fail loudly so a typo in config.yaml gets caught.
    raise ValueError(f"Unknown aggregation: {aggregation!r}. "
                     f"Choose from lik_gt_0 | lik_eq_100 | primary_max_lik.")


def compute_stft(signal: np.ndarray, fs: int = 100,
                 nfft: int = 480) -> np.ndarray:
    """
    Short-time Fourier transform (STFT) magnitude for ONE lead. This is the
    time-frequency view the model trains on instead of the raw trace: it shows
    how the ECG's frequency content evolves across the beat.

    Args:
        signal: 1-D samples for a single ECG lead (length 1000 at 100 Hz; the
                Dataset calls this once per lead).
        fs:     sampling rate in Hz, passed to scipy only to label its (unused)
                frequency axis -- the returned magnitudes do not depend on it.
        nfft:   FFT length applied to each (zero-padded) window. Sets the
                frequency resolution: F = nfft//2 + 1 output bins.

    Returns:
        A (F, T) float32 array of magnitudes, F = nfft//2 + 1, T fixed by the
        windowing (see below).

    STFT parameters from Section 4.3 of the original study:
    - nperseg=64:  window size
    - noverlap=48: hop = 16, so noverlap = 64 - 16 = 48 (75% overlap)
    - nfft=480:    zero-pads each window to 480 points before the FFT, giving
                   F = 480//2 + 1 = 241 frequency bins (the default)
    - T = 59 time frames for a 1000-sample 100 Hz trace (verified empirically;
      it depends on nperseg/noverlap, NOT on nfft, so T is invariant when nfft
      is swept)

    Larger nfft gives finer frequency resolution. Table 2 of the original study
    sweeps nfft over {240, 360, 480, 512}; 480 wins by a small margin.
    """
    # boundary=None, padded=False reproduce the framing of the original study
    # exactly: scipy does NOT pad or extend the signal at the edges, so the frame
    # count T is fixed purely by (len(signal) - noverlap) // (nperseg - noverlap)
    # and lands on the expected 59 frames. Flipping either flag would shift T and
    # break the (12, 241, 59) shape contract the model and reshape depend on.
    # I discard the returned frequency and time axes (f, t) -- only the complex
    # spectrogram Zxx is needed.
    _, _, Zxx = scipy_stft(signal, fs=fs, nperseg=64, noverlap=48,
                           nfft=nfft, boundary=None, padded=False)
    # np.abs collapses the complex STFT to magnitude (phase dropped, matching the
    # spectrogram input of the original study). Cast to float32 to match the
    # model dtype and halve the cache footprint vs float64.
    # NOTE: this is a linear-magnitude spectrogram; the original study applies no
    # log/dB compression here, so I do not either -- the model's input
    # normalization (in train.py) handles scaling instead.
    return np.abs(Zxx).astype(np.float32)


def load_signal(data_dir: str, filename: str, fs: int = 100) -> np.ndarray:
    """
    Load one PTB-XL record's 12-lead waveform via WFDB.

    Args:
        data_dir: root of the unpacked PTB-XL release.
        filename: record path RELATIVE to data_dir, taken from the filename_lr
                  (100 Hz) or filename_hr (500 Hz) column. It carries NO file
                  extension -- wfdb.rdsamp appends .hea/.dat itself.
        fs:       the sampling rate I expect; asserted against the header so a
                  filename/rate mismatch fails fast.

    Returns:
        An (n_samples, 12) float32 array of lead voltages -- (1000, 12) at
        100 Hz, (5000, 12) at 500 Hz. Lead/time order is whatever WFDB returns;
        the Dataset slices it column-wise per lead.

    Called by PTBXLDataset.__getitem__ only on a cache miss.
    """
    record_path = os.path.join(data_dir, filename)
    # rdsamp returns (signals ndarray, fields dict). fields["fs"] is the header's
    # declared sampling rate; I cross-check it against the rate implied by the
    # column the filename came from, so passing a 500 Hz filename while expecting
    # 100 Hz (or vice versa) is caught here rather than silently producing a
    # wrong-length STFT.
    signals, fields = wfdb.rdsamp(record_path)
    assert fields["fs"] == fs, (
        f"Expected fs={fs}, got {fields['fs']}. "
        f"Check that filename points to the correct sampling rate directory."
    )
    # wfdb yields float64; downcast to float32 to match compute_stft and the model.
    return signals.astype(np.float32)


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 5-superclass multi-label classification.

    Lifecycle:
      - __init__ does the metadata work once: load the CSVs, select the fold for
        'split', drop broken records, build every label vector, and keep only
        records that carry at least one superclass label. It then prints the
        per-class counts (a deliberate visibility guard against silent label
        drift).
      - __getitem__ does the heavy per-record work lazily: read the waveform,
        STFT each lead, stack to (12, F, T), and (if cache_dir is set) persist
        the spectrogram so later epochs are a fast np.load.

    Each item returns:
        x : (12, 241, 59) float32 tensor - per-lead STFT magnitude
                                           (241 = nfft//2+1 for the default nfft=480)
        y : (5,)          float32 tensor - binary multi-label vector (SUPERCLASSES order)

    Instantiated by train.py, evaluate_checkpoint.py, and threshold_tune.py.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,                       # "train", "val", or "test"
        fs: int = 100,
        cache_dir: str = None,
        use_cuda_backend: bool = False,
        label_aggregation: str = "lik_eq_100",
        nfft: int = 480,
    ):
        """
        Args:
            data_dir:          root of the unpacked PTB-XL release (holds the two
                               CSVs and the records100/ records500/ trees).
            split:             "train" | "val" | "test"; selects the strat_fold
                               group (see FOLD_TRAIN/VAL/TEST).
            fs:                100 (default, reads filename_lr) or 500 (reads
                               filename_hr). The whole pipeline is tuned for 100 Hz.
            cache_dir:         where per-record STFT .npy files live. If None,
                               every __getitem__ recomputes the STFT from the
                               waveform (correct but slow). Strongly recommended
                               to set and to pre-fill via precompute_stft.py.
            use_cuda_backend:  accepted for API symmetry with the other datasets
                               but unused here (the STFT runs on CPU in
                               numpy/scipy).
            label_aggregation: passed straight to build_label_vector; defaults to
                               the project's lik_eq_100 high-confidence rule.
            nfft:              STFT FFT length; sets F = nfft//2+1 and therefore
                               the model input size. Must match cache_dir's nfft.
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.data_dir          = data_dir
        self.cache_dir         = cache_dir
        self.fs                = fs
        self.label_aggregation = label_aggregation
        # NOTE: use_cuda_backend is deliberately not stored -- it exists only to
        # keep the constructor signature uniform across the PTB-XL and Georgia
        # datasets so callers can swap one for the other without changing kwargs.
        # nfft drives the STFT frequency resolution (F = nfft//2 + 1) and hence
        # the model's input_size = 12 * F. The caller must pass a cache_dir that
        # matches this nfft -- cache files hold (12, F, T) tensors whose shape
        # differs across nfft values.
        self.nfft = nfft

        # Load metadata (one row per record, plus the SCP-code dictionary).
        df = load_ptbxl_metadata(data_dir)
        scp_df = load_scp_statements(data_dir)

        # Build the SCP code -> superclass mapping. Keep only rows whose
        # diagnostic_class is set (NaN = non-diagnostic rhythm/form codes, which
        # carry no superclass), then collapse to a plain dict for O(1) lookup
        # inside build_label_vector.
        self.scp_to_super = (
            scp_df[scp_df["diagnostic_class"].notna()]["diagnostic_class"]
            .to_dict()
        )

        # Keep only the records in this split's folds. .copy() detaches the
        # slice from the parent frame so the later .loc assignment does not raise
        # a SettingWithCopy warning.
        fold_map = {"train": FOLD_TRAIN, "val": FOLD_VAL, "test": FOLD_TEST}
        folds = fold_map[split]
        df = df[df["strat_fold"].isin(folds)].copy()

        # Drop records with broken WFDB headers (see KNOWN_BROKEN_ECG_IDS).
        broken_in_split = sorted(set(df.index) & KNOWN_BROKEN_ECG_IDS)
        if broken_in_split:
            print(f"[PTBXLDataset] dropping {len(broken_in_split)} known-broken "
                  f"records from {split}: {broken_in_split}", flush=True)
            df = df.drop(index=broken_in_split)

        # Pick the column holding the record paths for the requested rate: the
        # high-resolution (500 Hz) tree vs the low-resolution (100 Hz) tree.
        filename_col = "filename_hr" if fs == 500 else "filename_lr"

        # Build label vectors and, in the same pass, drop any record whose label
        # is all-zero under the chosen aggregation (no usable superclass). I
        # accumulate into parallel python lists and stack once at the end --
        # cheaper than growing an ndarray row by row.
        labels = []
        valid_idx = []
        for ecg_id, row in df.iterrows():
            lv = build_label_vector(row["scp_codes"], self.scp_to_super,
                                     aggregation=label_aggregation)
            if lv.sum() > 0:                  # at least one positive superclass
                labels.append(lv)
                valid_idx.append(ecg_id)

        # df, labels, and filenames are now index-aligned: row i of self.labels
        # is the label for self.df.index[i], whose waveform path is
        # self.filenames[i]. __getitem__ depends on that alignment.
        self.df        = df.loc[valid_idx]
        self.labels    = np.stack(labels, axis=0)        # (N, 5) float32
        self.filenames = self.df[filename_col].values    # (N,) relative-path strings

        # Visibility: log the class counts at construction so silent label drift
        # is caught immediately. Invisible data/label issues have bitten this
        # project before -- print the numbers, every time.
        per_class = self.labels.sum(0).astype(int).tolist()
        print(f"[PTBXLDataset] split={split}  agg={label_aggregation}  "
              f"N={len(self.labels)}  per-class {dict(zip(SUPERCLASSES, per_class))}",
              flush=True)

    def __len__(self) -> int:
        """Number of usable records in this split, after fold filtering and
        dropping label-less records. Defines the index range for __getitem__."""
        return len(self.filenames)

    def __getitem__(self, idx: int):
        """
        Return the (x, y) pair for record idx in [0, len(self)).

        Returns:
            x: (12, F, T) float32 tensor -- per-lead STFT magnitude
               (F = nfft//2+1, T = 59 for 100 Hz).
            y: (5,) float32 tensor -- the multi-label superclass vector.

        Cache behavior: on a hit, load the precomputed spectrogram from
        {cache_dir}/{ecg_id}.npy; on a miss, compute it from the waveform and (if
        caching) write it atomically. There is deliberately NO "on error,
        substitute random" fallback -- any I/O or decode error propagates.
        """
        # Map the positional dataset index to the record's PTB-XL ecg_id, which
        # also names its cache file. self.df.index[idx] aligns with
        # self.labels[idx] and self.filenames[idx] by construction (see __init__).
        ecg_id = self.df.index[idx]
        cache_path = os.path.join(self.cache_dir, f"{ecg_id}.npy") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            # Fast path: spectrogram already on disk (pre-built or built in an
            # earlier epoch). This is the common case during training.
            x = np.load(cache_path)
        else:
            # No silent fallback: any failure here (missing file, unwritable
            # cache dir, wfdb error) must raise rather than quietly poison
            # training with random tensors.
            signal = load_signal(self.data_dir, self.filenames[idx], self.fs)  # (T_samp, 12)
            # STFT each lead on its own. signal[:, lead] is the 1-D trace for one
            # lead and compute_stft returns its (F, T) spectrogram; repeat for
            # all 12 leads. This list-comp over 12 leads is fine for a one-off
            # cache build, but it is why pre-warming the cache with
            # precompute_stft.py (multiprocess) matters for throughput.
            stft_leads = [compute_stft(signal[:, lead], self.fs, self.nfft)
                          for lead in range(signal.shape[1])]
            # Stack the 12 (F, T) lead spectrograms along a new leading axis ->
            # (12, F, T), the channel-first layout the model's reshape expects.
            x = np.stack(stft_leads, axis=0)
            if cache_path:
                # Persist the spectrogram to the cache (mirrors GeorgiaECGDataset)
                # so a lazily-built cache survives across epochs instead of
                # recomputing every record every epoch. Write to a unique temp
                # file then atomically rename, so concurrent DataLoader workers
                # can never read a partial .npy. Fails loudly if the cache dir is
                # unwritable.
                os.makedirs(self.cache_dir, exist_ok=True)
                # The per-process temp name keeps two DataLoader workers from
                # racing on the same path; os.replace is atomic on POSIX, so a
                # reader only ever sees either the old file or the complete new
                # one.
                tmp = f"{cache_path}.tmp.{os.getpid()}"
                try:
                    with open(tmp, "wb") as fh:
                        np.save(fh, x)
                    os.replace(tmp, cache_path)        # atomic publish
                except BaseException:
                    # Catch BaseException (not just Exception) so that even a
                    # KeyboardInterrupt/SystemExit mid-write still removes the
                    # half-written temp file before re-raising -- no orphan .tmp
                    # left to confuse a later run.
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
                # NOTE: a long-lived lazy cache can accumulate stale .tmp.<pid>
                #       files if a process is killed with SIGKILL (which
                #       BaseException cannot trap). A startup sweep deleting
                #       *.tmp.* in cache_dir would make the cache self-healing.

        # Hand back torch tensors viewing the numpy buffers (zero-copy). The
        # DataLoader's default collate stacks these into (B, 12, F, T) and (B, 5);
        # train.py then permutes/reshapes x to (B, T, 12*F) before the model.
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.labels[idx]),
        )