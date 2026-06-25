"""
tests/test_data.py
==================
Contract tests for the data layer of the xLSTM-ECG reproduction.

Every PTB-XL AUROC I report traces back to one decision -- how
build_label_vector aggregates SCP codes into a (5,) superclass vector -- and
every spectrogram shape traces back to compute_stft's nfft. A typo or an
accidental default change in either place would not crash; it would quietly
relabel records or reshape features and hand me a different, wrong number.

  * the 5 PTB-XL superclasses and their fixed order,
  * the three label-aggregation strategies (lik_gt_0, lik_eq_100,
    primary_max_lik) and their tie-breaking,
  * the STFT magnitude shape (F, T) with F = nfft//2 + 1, T = 59, across the
    N_FFT sweep values used in the original study,
  * and that evaluate_checkpoint / threshold_tune pass the training-time label
    and feature knobs through to the dataset unchanged.

How it is run
-------------
    pytest                                  # whole suite (config in pytest.ini)
    pytest tests/test_data.py               # just this file
    pytest tests/test_data.py::test_compute_stft_default_nfft_is_480   # one test

"""
import numpy as np
import pytest
# The three things under test, imported from the PTB-XL loader. SUPERCLASSES is
# the single source of truth for label-column order; build_label_vector and
# compute_stft are the two functions that define a label and a feature tensor,
# respectively. Importing by package path (src.data.dataset) relies on the
# sys.path fix-up in conftest.py / the 'pythonpath = .' in pytest.ini.
from src.data.dataset import (
    build_label_vector,
    compute_stft,
    SUPERCLASSES,
)


def test_superclasses_length():
    """The PTB-XL label space is exactly 5 diagnostic superclasses
    (NORM, MI, STTC, CD, HYP). The whole pipeline -- label-vector width, model
    num_classes, per-class metric tables -- is sized off this constant, so a
    change to its length is a breaking change."""
    assert len(SUPERCLASSES) == 5


def test_build_label_vector_norm():
    """Default aggregation (lik_gt_0), single confident NORM code.

    NORM is position 0 and the vector has the fixed width 5. scp_to_super
    deliberately includes extra codes (IMI->MI, NDT->STTC) that are not present
    in the record, to confirm the mapping is consulted by membership, not
    blindly applied to whatever it contains."""
    scp_to_super = {"NORM": "NORM", "IMI": "MI", "NDT": "STTC"}
    lv = build_label_vector({"NORM": 100.0}, scp_to_super)
    assert lv[0] == 1.0  # NORM fires (mapped to superclass NORM at index 0)
    assert lv[1] == 0.0  # MI stays 0 -- no MI code was present in this record
    assert lv.shape == (5,)  # always 5-wide regardless of how many codes fired


def test_build_label_vector_multilabel():
    """Two codes mapping to two different superclasses must produce a genuine
    multi-label vector (sum == 2), not a single-label pick.

    This is the defining property of lik_gt_0 / lik_eq_100: one ECG can be
    positive for several superclasses at once, which is clinically the norm --
    an infarct and a conduction problem coexist all the time. IMI->MI (index 1)
    and LBBB->CD (index 3) both fire here. Note LBBB maps to CD (conduction
    disturbance); BBB would be a better superclass."""
    scp_to_super = {"IMI": "MI", "LBBB": "CD"}
    lv = build_label_vector({"IMI": 80.0, "LBBB": 100.0}, scp_to_super)
    assert lv[1] == 1.0  # MI  (from IMI, lik=80 -> still > 0)
    assert lv[3] == 1.0  # CD  (from LBBB, lik=100)
    assert lv.sum() == 2.0  # exactly two positives, nothing else flipped


def test_build_label_vector_unknown_code():
    """A code that is not in scp_to_super contributes nothing -- the result is
    the all-zero vector.

    This is the 'no usable label' case. Upstream, PTBXLDataset.__init__ drops
    records whose label vector sums to 0, so this behavior is what makes
    non-diagnostic / unmapped codes get filtered out rather than mislabeled."""
    scp_to_super = {"NORM": "NORM"}
    lv = build_label_vector({"UNKNOWNCODE": 100.0}, scp_to_super)
    assert lv.sum() == 0.0  # nothing mapped -> record would be dropped upstream


# -- Aggregation contract tests (locked 2026-05-10 with the
#    label_aggregation rewrite). --------------------------------------

def test_lik_gt_0_excludes_zero_likelihood():
    """Under lik_gt_0, only entries with lik > 0 contribute. PTB-XL records can
    carry codes with lik=0.0 -- the cardiologist considered the diagnosis and
    explicitly ruled it out (NOTE: Could test initial labelling methods in PTB-XL?)
    -- and the aggregator must drop those, not count them as present."""
    scp_to_super = {"NORM": "NORM", "IMI": "MI"}
    # IMI is present with lik=0.0 -- considered but explicitly not present.
    # The strict '> 0' test (not '>=') is the thing being pinned: lik=0 must NOT
    # turn the MI bit on.
    lv = build_label_vector({"NORM": 100.0, "IMI": 0.0}, scp_to_super,
                             aggregation="lik_gt_0")
    assert lv[0] == 1.0  # NORM (lik=100, clearly > 0)
    assert lv[1] == 0.0  # MI was lik=0, must be excluded by the '> 0' rule
    assert lv.sum() == 1.0  # exactly one positive -> the lik=0 code added nothing


def test_lik_eq_100_only_high_confidence():
    """Under lik_eq_100, only codes the cardiologist was fully certain of
    (lik exactly 100) contribute. Hedged codes (lik=80, 50, ...) are dropped."""
    scp_to_super = {"NORM": "NORM", "IMI": "MI", "LBBB": "CD"}
    # IMI at lik=80 is the discriminating case: under lik_gt_0 it would fire MI,
    # but under lik_eq_100 it is dropped. This is exactly the train/eval-mismatch
    # bug guarded elsewhere -- the two aggregations give different ground truth.
    lv = build_label_vector({"NORM": 100.0, "IMI": 80.0, "LBBB": 100.0},
                             scp_to_super, aggregation="lik_eq_100")
    assert lv[0] == 1.0  # NORM (lik exactly 100)
    assert lv[1] == 0.0  # MI (lik=80 -> below threshold, dropped)
    assert lv[3] == 1.0  # CD (LBBB at lik exactly 100)
    assert lv.sum() == 2.0  # only the two lik==100 codes survive


def test_primary_max_lik_picks_highest():
    """Under primary_max_lik, exactly one class fires -- the superclass of the
    code with the highest likelihood."""
    scp_to_super = {"IMI": "MI", "LBBB": "CD", "NORM": "NORM"}
    # Three competing codes at 50 / 100 / 80; LBBB (lik=100) is the unique max,
    # so CD (index 3) is the single winner. sum==1 confirms single-label output:
    # this aggregation forces the most-confident reading to win outright.
    lv = build_label_vector({"IMI": 50.0, "LBBB": 100.0, "NORM": 80.0},
                             scp_to_super, aggregation="primary_max_lik")
    assert lv[3] == 1.0    # CD wins (lik=100 is the unique maximum)
    assert lv.sum() == 1.0  # single-label: exactly one bit set


def test_primary_max_lik_ties_broken_by_superclass_order():
    """Ties in likelihood resolve by SUPERCLASSES order (NORM, MI, STTC, CD,
    HYP). NORM beats MI on a tie."""
    scp_to_super = {"NORM": "NORM", "IMI": "MI"}
    # Both at lik=100 -> a genuine tie. The secondary sort key in
    # build_label_vector is SUPERCLASSES.index, so the earlier-listed superclass
    # (NORM, index 0) deterministically wins. This makes the labeling
    # reproducible regardless of dict iteration order/
    # NOTE: Not physiologically inspired - ties should be broken differently.
    lv = build_label_vector({"NORM": 100.0, "IMI": 100.0}, scp_to_super,
                             aggregation="primary_max_lik")
    assert lv[0] == 1.0    # NORM wins on tie (earlier in SUPERCLASSES than MI)
    assert lv.sum() == 1.0  # still single-label even with a tie
    # This only pins the NORM-vs-MI ordering. A fuller guard could parametrize
    # over every adjacent SUPERCLASSES pair, but NORM-over-MI is the tie that
    # actually occurs in PTB-XL (a healthy-looking read co-annotated with MI),
    # so it is the one worth locking.


def test_unknown_aggregation_raises():
    """An unrecognized aggregation string must raise ValueError, not silently
    return the all-zero vector. A typo in config.yaml (e.g. 'lik_eq100') would
    otherwise zero out every label and 'train' on nothing -- so this failure has
    to be loud. The match= pins the error-message prefix so the contract also
    covers which error is raised, not just that something raised."""
    scp_to_super = {"NORM": "NORM"}
    with pytest.raises(ValueError, match="Unknown aggregation"):
        build_label_vector({"NORM": 100.0}, scp_to_super,
                           aggregation="not_a_real_strategy")

def test_compute_stft_shape():
    """The single-lead STFT magnitude must come out as (241, 59) float32 for the
    default nfft=480 at 100 Hz on a 1000-sample (10 s) signal.

    This is THE shape contract the whole model rests on: 12 leads x 241 freq
    bins x 59 frames, later reshaped to (B, 59, 12*241=2892) before the xLSTM.
      * F = nfft//2 + 1 = 241 (real-FFT one-sided bin count),
      * T = 59 frames, fixed by nperseg=64 / noverlap=48 with scipy's
        boundary=None, padded=False framing -- it depends on the windowing."""
    signal = np.random.randn(1000).astype(np.float32)  # 10 s @ 100 Hz, one lead
    mag = compute_stft(signal, fs=100)  # default nfft=480 (Section 4.3 of the original study)
    # nfft=480 -> F = 480//2 + 1 = 241 frequency bins (one-sided real FFT)
    assert mag.shape[0] == 241
    # actual frames from scipy with boundary=None, padded=False: 59
    # (= (1000 - noverlap)//(nperseg - noverlap) = (1000-48)//16, capped by framing)
    assert mag.shape[1] == 59
    assert mag.dtype == np.float32  # float32, not float64 -- matches model dtype & halves cache size

def test_compute_stft_nonnegative():
    """STFT magnitude is |Zxx|, so every entry must be >= 0 by construction.

    A negative value here would mean someone returned the real part or a signed
    spectrum instead of the magnitude."""
    signal = np.random.randn(5000).astype(np.float32)
    mag = compute_stft(signal, fs=500)
    assert (mag >= 0).all()  # |Zxx| is non-negative everywhere


# -- nfft contract tests (locked 2026-05-10 with the Table 2 N_FFT
#    sweep infrastructure).

def test_compute_stft_nfft_240_shape():
    """nfft=240 -> F = 240//2 + 1 = 121 frequency bins. T unchanged at 59."""
    signal = np.random.randn(1000).astype(np.float32)
    mag = compute_stft(signal, fs=100, nfft=240)
    # F shrinks with nfft (121 here) but T stays 59 -- T is set by the windowing
    # (nperseg/noverlap).
    assert mag.shape == (121, 59)
    assert mag.dtype == np.float32


def test_compute_stft_nfft_360_shape():
    """nfft=360 -> F = 181."""
    signal = np.random.randn(1000).astype(np.float32)
    mag = compute_stft(signal, fs=100, nfft=360)
    assert mag.shape == (181, 59)  # 360//2 + 1 = 181 freq bins; T still 59


def test_compute_stft_nfft_512_shape():
    """nfft=512 -> F = 257."""
    signal = np.random.randn(1000).astype(np.float32)
    mag = compute_stft(signal, fs=100, nfft=512)
    assert mag.shape == (257, 59)  # 512//2 + 1 = 257 freq bins; T still 59


def test_compute_stft_default_nfft_is_480():
    """When nfft is not passed, the function must default to nfft=480
    (Section 4.3 of the original study) -- F = 241. """
    signal = np.random.randn(1000).astype(np.float32)
    # Crucially, no nfft passed. This pins the default, which is what every
    # cached spectrogram and the model's input_size=2892 (=12*241) assume. If
    # the signature default drifts, this is the test that fails.
    mag = compute_stft(signal, fs=100)
    assert mag.shape == (241, 59)  # 480//2 + 1 = 241 == the original study's default


# -- Eval-script wiring contract tests (locked 2026-05-10). Verifies that the eval scripts
#    construct datasets with the correct kwargs from a run_cfg snapshot,
#    so the bug class "model trained on lik_eq_100, evaluated on
#    lik_gt_0" cannot recur. My bad, but it happened.

def test_evaluate_checkpoint_threads_label_aggregation_through(monkeypatch):
    """evaluate_checkpoint.load_test_dataset must pass label_aggregation from
    the run_cfg snapshot to PTBXLDataset.

    Args:
        monkeypatch: pytest fixture used to swap in the mock dataset class and
            auto-restore the real one when the test finishes."""

    from scripts import evaluate_checkpoint
    captured = {}  # the mock writes the constructor kwargs here for inspection

    class _MockPTBXLDataset:
        # Accept-anything constructor: it does no real loading, it just snapshots
        # every kwarg load_test_dataset passed so I can assert on them below.
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Patch the import target inside load_test_dataset's local-import scope.
    # load_test_dataset does 'from src.data.dataset import PTBXLDataset' at call
    # time, so the name it resolves is src.data.dataset.PTBXLDataset -- that is
    # the exact attribute I override here. Patching a re-export on the scripts
    # module instead would miss it.
    monkeypatch.setattr(
        "src.data.dataset.PTBXLDataset",
        _MockPTBXLDataset,
    )

    # A minimal stand-in for a result.json config snapshot. data_dir/cache_dir
    # are fake paths -- they are never opened because the mock does no I/O.
    run_cfg = {
        "dataset_type":      "ptbxl",
        "data_dir":          "/fake",
        "cache_dir":         "/fake_cache",
        "label_aggregation": "lik_eq_100",   # the value that MUST be threaded
        "nfft":              480,            # ditto (also defines feature shape)
    }
    evaluate_checkpoint.load_test_dataset(run_cfg)  # builds the (mock) test set

    # The two assertions that matter: both training-time knobs reached the
    # constructor verbatim. int(...) mirrors the production cast that tolerates
    # nfft arriving as a string from a JSON snapshot.
    assert captured.get("label_aggregation") == "lik_eq_100"
    assert int(captured.get("nfft", -1)) == 480


def test_evaluate_checkpoint_threads_georgia_knobs_through(monkeypatch):
    """For Georgia, both georgia_split_strategy and
    georgia_drop_no_target_codes must be threaded through.


    Args:
        monkeypatch: swaps in the mock Georgia dataset and restores it after."""
    from scripts import evaluate_checkpoint
    captured = {}

    class _MockGeorgiaECGDataset:
        # Same capture-only stub pattern as the PTB-XL test.
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Patch the Georgia loader at its real module path -- load_test_dataset
    # imports it locally as 'from src.data.georgia_dataset import GeorgiaECGDataset'.
    monkeypatch.setattr(
        "src.data.georgia_dataset.GeorgiaECGDataset",
        _MockGeorgiaECGDataset,
    )

    run_cfg = {
        "dataset_type":                  "georgia",
        "data_dir":                      "/fake",
        "cache_dir":                     "/fake_cache",
        "georgia_split_strategy":        "paper_strict",
        "georgia_drop_no_target_codes":  False,
    }
    evaluate_checkpoint.load_test_dataset(run_cfg)

    # Prefix stripped (georgia_split_strategy -> split_strategy) and value intact.
    assert captured.get("split_strategy") == "paper_strict"
    # 'is False', not '== False': this also rejects values like 0
    # or None, confirming the production bool(...) cast produced a real bool.
    assert captured.get("drop_no_target_codes") is False


def test_threshold_tune_threads_label_aggregation_through(monkeypatch):
    """threshold_tune._load_split must pass label_aggregation through for
    PTB-XL.

    Args:
        monkeypatch: swaps in the mock PTBXLDataset, restored after the test."""
    from scripts import threshold_tune
    captured = {}

    class _MockPTBXLDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # _load_split also does a local 'from src.data.dataset import PTBXLDataset',
    # so patching the source module's attribute is what intercepts the call.
    monkeypatch.setattr(
        "src.data.dataset.PTBXLDataset",
        _MockPTBXLDataset,
    )

    run_cfg = {
        "dataset_type":      "ptbxl",
        "data_dir":          "/fake",
        "cache_dir":         "/fake_cache",
        "label_aggregation": "lik_eq_100",
        "nfft":              512,           # non-default on purpose (see docstring)
    }
    # split="test" is threaded straight to the (mock) dataset; the threshold
    # tuner uses "train" to calibrate and "test" to report, but here I only care
    # that the label/feature knobs ride along -- any valid split exercises that.
    threshold_tune._load_split(run_cfg, split="test")

    assert captured.get("label_aggregation") == "lik_eq_100"
    assert int(captured.get("nfft", -1)) == 512  # the passed 512, not the 480 default