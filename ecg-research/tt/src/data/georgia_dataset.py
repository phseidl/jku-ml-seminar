"""
src/data/georgia_dataset.py
===========================
Georgia 12-Lead ECG Challenge dataset (PhysioNet/CinC 2020, G12EC subset).

Source:
    Perez Alday EA et al. "Classification of 12-lead ECGs: the PhysioNet/Computing
    in Cardiology Challenge 2020." Physiol Meas. 2020.
    DOI: https://doi.org/10.1088/1361-6579/abc960

"""

# NOTE: this module mixes os.path and pathlib.Path. Newer project code prefers
# pathlib, but the on-disk record walk below predates that convention and is
# left as-is to avoid churning behavior I have already verified; only the
# cache-key handling uses Path.
import os
from pathlib import Path

import numpy as np
import torch
# resample_poly: polyphase (anti-aliased) resampler for the 500->100 Hz
#                downsample; scipy_stft: short-time Fourier transform that turns
#                each 1-D lead signal into a time-frequency magnitude map.
from scipy.signal import resample_poly
from scipy.signal import stft as scipy_stft
from torch.utils.data import Dataset

# Class names in label-vector order (index 0..6). This list is the single
# source of truth for the Georgia label ordering: GEORGIA_CLASSES[i] names the
# diagnosis whose position is column i of the (7,) label vector, and every
# consumer (train.py, evaluate_checkpoint.py, threshold_tune.py) imports it to
# label per-class metrics. Reorder it and every reported number is silently
# attached to the wrong diagnosis.
GEORGIA_CLASSES = ["NSR", "AF", "IAVB", "LBBB", "RBBB", "SB", "STach"]

# SNOMED-CT code (the literal string found in the .hea #Dx line) -> class
# index. Keys are strings, not ints, because the header values are parsed as
# text and never converted. A code absent from this map is simply ignored --
# that is how the "only these 7 diagnoses count" rule is enforced; everything
# else the cardiologist coded is dropped.
SNOMED_TO_IDX: dict[str, int] = {
    "426783006": 0,   # NSR
    "164889003": 1,   # AF
    "270492004": 2,   # IAVB
    "164909002": 3,   # LBBB
    "59118001":  4,   # RBBB
    "426177001": 5,   # SB
    "427084000": 6,   # STach
}

# Derived once so the label-vector width and the class list can never drift
# apart: the (7,) vectors in _build_label and the per-class print all key off
# this one count.
NUM_CLASSES = len(GEORGIA_CLASSES)

# Group subfolders in data_dir; two split strategies are supported.
#
# "default": three-way split with g2 as a held-out validation set -- it lets
#            the train loop track val AUROC, do early stopping, and pick the
#            best epoch. This is the clinically sensible setup: model selection
#            happens on records the model never trained on.
#
# "paper_strict": the exact split from Section 4.7 of the original study -- g1
#            is the test set, all of g2..g11 are used for training, and there is
#            no separate validation set. To keep the existing train loop happy
#            (it always builds a val loader for per-epoch monitoring) the val
#            split just re-uses the training records; val AUROC is therefore NOT
#            a generalization signal in this mode and must not drive early
#            stopping or model selection. Pair it with patience large enough
#            that early stopping never fires (e.g. patience=999 and
#            num_epochs=20, matching the original study).
SPLIT_STRATEGIES: dict[str, dict[str, list[str]]] = {
    "default": {
        "test":  ["g1"],
        "val":   ["g2"],
        "train": ["g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11"],
    },
    "paper_strict": {
        "test":  ["g1"],
        "val":   ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11"],
        "train": ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11"],
    },
}

# Backwards compatibility for callers that still import _SPLIT_GROUPS. This
# private name predates SPLIT_STRATEGIES (back when only the "default" 3-way
# split existed); it still points at the default mapping so older imports keep
# working.
# NOTE: once no caller imports _SPLIT_GROUPS, drop this alias and have callers
#       read SPLIT_STRATEGIES["default"] directly.
_SPLIT_GROUPS = SPLIT_STRATEGIES["default"]


def _parse_snomed_codes(hea_path: str) -> list[str]:
    """Return the SNOMED-CT code strings on the '#Dx' line of a WFDB '.hea'
    header -- i.e. the diagnoses the original reader assigned to this recording.

    The Georgia headers encode the diagnoses as one comma-separated SNOMED-CT
    list on a line that looks like '#Dx: 426783006,164889003'. I scan line by
    line and stop at the first 'Dx' line.

    Parameters
    ----------
    hea_path : str
        Path to the '.hea' header file for one recording.

    Returns
    -------
    list[str]
        The raw code strings exactly as written in the header (whitespace
        trimmed). Empty list if no '#Dx' line is present.

    Notes
    -----
    Codes are returned as strings, never ints, so they line up with the string
    keys of 'SNOMED_TO_IDX' without any conversion.
    """
    with open(hea_path, "r") as fh:
        for line in fh:
            # Strip the leading '#' marker(s) and any following space so a
            # header written as "# Dx:" matches the same way as "#Dx:".
            stripped = line.lstrip("#").lstrip()
            # Case-insensitive prefix test: real headers use "Dx:", but I do not
            # want to be tripped up by casing.
            if stripped.lower().startswith("dx:"):
                # Take everything after the FIRST colon (maxsplit=1) so a stray
                # colon inside the code list would not truncate it.
                raw = stripped.split(":", 1)[1]
                # Split on commas, trim each token, and drop empties (handles a
                # trailing comma or "code, , code" gracefully).
                return [c.strip() for c in raw.split(",") if c.strip()]
    # No #Dx line found -> caller treats this as "no target codes present".
    return []


def _build_label(snomed_codes: list[str]) -> np.ndarray:
    """Turn a list of SNOMED-CT code strings into a (7,) binary float32 label.

    Multi-label by design, which matches the clinic: one ECG can be coded with
    several of the 7 diagnoses at once (e.g. atrial fibrillation with a bundle
    branch block), so every matching code sets its own column to 1.0 and the
    result may have more than one positive entry -- or none, if the record
    carried no diagnosis among the 7.

    Parameters
    ----------
    snomed_codes : list[str]
        Code strings as returned by _parse_snomed_codes.

    Returns
    -------
    np.ndarray
        A length-7 float32 vector. float32 (not bool/int) so it drops straight
        into a 'torch' tensor as a BCE target with no dtype cast.
    """
    # Start all-zero; only the 7 target codes flip a bit to 1.0.
    label = np.zeros(NUM_CLASSES, dtype=np.float32)
    for code in snomed_codes:
        # Codes outside the 7 targets return None and are skipped -- this is
        # what restricts the problem to the 7 challenge diagnoses.
        idx = SNOMED_TO_IDX.get(code)
        if idx is not None:
            label[idx] = 1.0
    return label


def _compute_stft_lead(signal_1d: np.ndarray, fs: int = 100) -> np.ndarray:
    """Compute the STFT magnitude map for a single ECG lead.

    The STFT parameters match 'PTBXLDataset' exactly so a Georgia sample and a
    PTB-XL sample have the same geometry and one model can consume either. For a
    1000-sample (10 s at 100 Hz) input, nperseg=64 / noverlap=48 (hop 16) /
    nfft=480 give nfft//2 + 1 = 241 frequency bins and 59 time frames.

    Parameters
    ----------
    signal_1d : np.ndarray
        One lead's time-domain signal, shape (1000,) at 'fs' Hz.
    fs : int, default 100
        Sampling rate in Hz (only used by scipy to label the frequency axis;
        the magnitude values do not depend on it).

    Returns
    -------
    np.ndarray
        (241, 59) float32 STFT magnitude (= |Zxx|).
    """
    # boundary=None, padded=False: do NOT zero-extend the signal at the edges.
    # This is what pins the time-frame count at exactly 59 for a 1000-sample
    # input; scipy's defaults would pad and yield a different frame count, which
    # would break the (12, 241, 59) shape contract shared with PTB-XL.
    _, _, Zxx = scipy_stft(
        signal_1d, fs=fs, nperseg=64, noverlap=48,
        nfft=480, boundary=None, padded=False,
    )
    # Keep magnitude only (drop phase) and cast to float32: halves the cache
    # size and matches the dtype the model expects.
    return np.abs(Zxx).astype(np.float32)


class GeorgiaECGDataset(Dataset):
    """
    PyTorch Dataset for the Georgia 12-Lead ECG Challenge data.

    Each item returns:
        x : (12, 241, 59) float32 tensor  - STFT magnitude per lead
        y : (7,)          float32 tensor  - binary multi-label vector

    Parameters
    ----------
    data_dir       : path containing g1/ ... g11/ subdirectories
    split          : "train" | "val" | "test"
    cache_dir      : directory for .npy STFT cache (recommended; None = no cache)
    split_strategy : "default" (g1 test, g2 val, g3-g11 train) or
                     "paper_strict" (g1 test, g2-g11 train, no held-out val).
                     See SPLIT_STRATEGIES for details.
    drop_no_target_codes : True (default) drops records carrying none of the 7
                     target SNOMED codes (the historical behavior; ~40.8% of
                     Georgia records are dropped this way -- they were coded
                     with some other diagnosis). False keeps them as all-zero
                     ("negative for all 7") label vectors. Section 4.7 of the
                     original study is silent on this choice, so it is exposed
                     as a knob rather than hard-coded: True matches my earlier
                     reproductions, False tests the alternative reading that the
                     original study kept every record.
    """

    def __init__(self, data_dir: str, split: str, cache_dir: str = None,
                 split_strategy: str = "default",
                 drop_no_target_codes: bool = True):
        """Index the on-disk records for one split (loads no signal).

        Construction is cheap and eager: it walks the relevant group folders,
        parses every '.hea' header, builds the label vector, and decides
        membership (kept vs. dropped). The signals/STFTs are loaded lazily in
        '__getitem__', so building all three splits up front is fast.

        See the class docstring for the meaning of every parameter.
        """
        # Fail fast on a bad split/strategy name rather than silently building
        # an empty dataset that would only surface later as a baffling 0-sample
        # run.
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        assert split_strategy in SPLIT_STRATEGIES, (
            f"Unknown split_strategy: {split_strategy!r}. "
            f"Must be one of: {sorted(SPLIT_STRATEGIES)}"
        )
        self.data_dir              = data_dir
        self.cache_dir             = cache_dir
        self.split_strategy        = split_strategy
        self.drop_no_target_codes  = drop_no_target_codes

        # Each entry holds everything __getitem__ needs without re-touching the
        # header: the group (for the cache subdir), the .mat path (for loading),
        # and the precomputed label (so the header is never re-parsed at train
        # time).
        records: list[tuple[str, str, np.ndarray]] = []  # (group, mat_path, label)
        n_kept_with_zero = 0   # counted only to print below

        # Resolve which group folders belong to this split under this strategy.
        groups = SPLIT_STRATEGIES[split_strategy][split]
        for group in groups:
            group_dir = os.path.join(data_dir, group)
            # Tolerate a missing group folder (e.g. a partial download) by
            # skipping it rather than crashing -- the per-class print at the end
            # will flag an unexpectedly small N.
            if not os.path.isdir(group_dir):
                continue
            # sorted() makes the record order determinstic across machines and
            # filesystems, so the cache layout and any index-based debugging
            # stay reproducible.
            for fname in sorted(os.listdir(group_dir)):
                # Drive the walk off the .hea headers; each header has a sibling
                # .mat signal file.
                if not fname.endswith(".hea"):
                    continue
                stem     = fname[:-4]                       # strip ".hea"
                hea_path = os.path.join(group_dir, fname)
                mat_path = os.path.join(group_dir, stem + ".mat")
                # Skip a header whose signal file is missing rather than letting
                # it blow up later inside a DataLoader worker.
                if not os.path.exists(mat_path):
                    continue
                codes = _parse_snomed_codes(hea_path)
                label = _build_label(codes)
                # label.sum() == 0 means none of this record's diagnoses are
                # among the 7 target classes. Under the default drop rule it is
                # excluded entirely; otherwise it is kept as an all-zero
                # ("negative for all 7") target and tallied for the audit line.
                if label.sum() == 0:
                    if drop_no_target_codes:
                        continue
                    n_kept_with_zero += 1
                records.append((group, mat_path, label))
        # One-line audit so a "keep all records" run is unmistakable in the log
        # (prints only when records were actually kept under that mode).
        if not drop_no_target_codes and n_kept_with_zero:
            print(f"[GeorgiaECGDataset] split={split}  kept {n_kept_with_zero} "
                  f"records with all-zero label vectors (drop_no_target_codes=False)",
                  flush=True)

        self._records = records
        # Stack the per-record labels into one (N, 7) matrix. Exposed publicly
        # because consumers (pos_weight computation, class-balance checks) want
        # the whole label matrix without iterating __getitem__ and loading every
        # signal.
        self.labels   = np.stack([r[2] for r in records], axis=0)  # (N, 7)

        # Print per-class counts at construction so silent label drift (a
        # changed SNOMED mapping, a changed split) is caught the moment it
        # happens rather than after a wasted sweep. These counts also let me eye
        # the class imbalance -- NSR and the brady/tachy classes dominate.
        # sum(0) collapses the (N, 7) matrix to a (7,) per-class positive count;
        # cast to int + tolist() keeps the printed dict clean (no numpy types).
        per_class = self.labels.sum(0).astype(int).tolist()
        print(f"[GeorgiaECGDataset] split={split}  strategy={split_strategy}  "
              f"N={len(self.labels)}  per-class {dict(zip(GEORGIA_CLASSES, per_class))}",
              flush=True)

    def __len__(self) -> int:
        """Number of recordings in this split (the unit the DataLoader iterates)."""
        return len(self._records)

    def __getitem__(self, idx: int):
        """Return one '(x, y)' sample, loading or caching its STFT on demand.

        Parameters
        ----------
        idx : int
            Position into this split's record list (0 .. len-1).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            'x' : (12, 241, 59) float32 STFT magnitude (leads, freq, time).
            'y' : (7,) float32 multi-label target.

        Notes
        -----
        The first access to a record computes the STFT and (if 'cache_dir' is
        set) writes it to disk atomically; every later access reads the cached
        '.npy' directly. The cache layout mirrors the data layout
        ('cache_dir/<group>/<stem>.npy') so two groups with same-named stems
        never collide.
        """
        group, mat_path, label = self._records[idx]
        stem = Path(mat_path).stem   # filename without directory or extension

        # Build the cache path, preserving the group subdirectory.
        # cache_path stays None when caching is disabled, which short-circuits
        # every cache branch below.
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, group, f"{stem}.npy")

        if cache_path and os.path.exists(cache_path):
            # Cache hit: load the precomputed (12, 241, 59) array straight off
            # disk -- no WFDB read, no resample, no STFT.
            x = np.load(cache_path)
        else:
            # Cache miss (or caching disabled): do the full load->resample->STFT.
            x = self._load_and_preprocess(mat_path)
            if cache_path:
                # Fail loudly if the cache directory is unwritable. A silent
                # fallback to random tensors here once masked a permission error
                # and quietly trained the model on noise -- never again.
                # Write to a unique temp file, then atomically rename, so a
                # concurrent DataLoader worker can never read a partial .npy.
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # PID-suffixed temp name keeps concurrent workers from clobbering
                # each other's in-flight writes for the same record.
                tmp = f"{cache_path}.tmp.{os.getpid()}"
                try:
                    with open(tmp, "wb") as fh:
                        np.save(fh, x)
                    # os.replace is atomic on the same filesystem: a reader sees
                    # either the old absence or the complete file, never a
                    # half-written one.
                    os.replace(tmp, cache_path)
                except BaseException:
                    # Catch BaseException (not just Exception) so even a
                    # KeyboardInterrupt / SystemExit mid-write still cleans up
                    # the temp file before re-raising -- no orphaned .tmp files.
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise

        # torch.from_numpy shares memory with the numpy array (no copy); both
        # are already float32, matching the model's input/target dtype.
        return torch.from_numpy(x), torch.from_numpy(label)

    def _load_and_preprocess(self, mat_path: str) -> np.ndarray:
        """Turn one raw WFDB recording into the model-ready STFT tensor.

        Pipeline (mirrors 'PTBXLDataset' so the two datasets are
        interchangeable downstream):

        1. 'wfdb.rdsamp'     -> (n_samples, 12) float32 at 500 Hz
        2. 'resample_poly'   -> 100 Hz (n_samples / 5 rows)
        3. crop/pad          -> exactly (1000, 12) (10 s at 100 Hz)
        4. per-lead STFT     -> (12, 241, 59) magnitude

        Parameters
        ----------
        mat_path : str
            Path to the '.mat' signal file (the sibling '.hea' is read by 'wfdb'
            automatically from the same stem).

        Returns
        -------
        np.ndarray
            (12, 241, 59) float32 STFT magnitude, ready to cache and to be
            wrapped in a tensor by '__getitem__'.
        """
        import wfdb

        # wfdb addresses a record by its stem (no extension); it reads both the
        # .mat data and the .hea header from that stem.
        stem    = mat_path[:-4]                          # strip .mat extension
        signals, fields = wfdb.rdsamp(stem)              # (n_samples, 12)
        # fields (sample rate, units, ...) is intentionally unused: 500 Hz is a
        # fixed property of this dataset, so the downsample factor is hard-coded
        # rather than read off fields["fs"].
        signals = signals.astype(np.float32)

        # Downsample 500 Hz -> 100 Hz (factor 5)
        # resample_poly avoids the phase distortion of plain decimation and
        # applies the proper anti-aliasing polyphase filter. up=1, down=5 gives
        # the exact 5x decimation along axis=0 (time), leaving the 12 lead
        # columns intact.
        signals_100hz = resample_poly(
            signals, up=1, down=5, axis=0,
        ).astype(np.float32)

        # Force exactly 1000 samples (10 s at 100 Hz) to match PTB-XL geometry.
        # Resampling can land a sample or two off 1000 (the source clip is not
        # always exactly 5000 samples), so the length is normalized here; the
        # downstream STFT shape (12, 241, 59) depends on this being exactly 1000.
        target = 1000
        n = signals_100hz.shape[0]
        if n > target:
            # Too long: keep the leading 1000 samples (drop the tail).
            signals_100hz = signals_100hz[:target, :]
        elif n < target:
            # Too short: zero-pad the END only (pad the time axis, not the
            # leads). np.pad's default mode is constant-zero, the neutral choice
            # for a magnitude STFT.
            # NOTE: center-padding would also work, but tail-padding keeps
            # parity with the PTB-XL loader and the real ECG onset always sits
            # at t=0.
            pad = target - n
            signals_100hz = np.pad(signals_100hz, ((0, pad), (0, 0)))

        # Per-lead STFT -> (12, 241, 59)
        # Run each of the 12 leads through the STFT independently (each lead ->
        # a (241, 59) map) and stack on a new leading axis to get
        # (leads, freq, time). shape[1] is the lead count (12); iterating it
        # rather than hard-coding 12 keeps the loop consistent with the array.
        stft_leads = [
            _compute_stft_lead(signals_100hz[:, lead], fs=100)
            for lead in range(signals_100hz.shape[1])
        ]
        return np.stack(stft_leads, axis=0)