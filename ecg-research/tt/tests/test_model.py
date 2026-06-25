"""
tests/test_model.py
===================
CPU-only contract tests for 'XLSTMECGModel' (the network in
'src/models/xlstm_ecg.py').

A pytest module that pins the model's output contract -- the three facts the
rest of the training and evaluation code quietly trusts:

  1. Shape: a batch of '(B, T, input_size)' STFT features comes back as
     '(B, num_classes)'. If this drifts, every downstream 'compute_metrics'
     call (AUROC, confusion matrix) either breaks or, worse, silently lines up
     the wrong class with the wrong probability and reports a plausible-looking
     number that is wrong.
  2. Range: every output sits in '[0, 1]'. The model ends in a sigmoid, so each
     class gets its own independent probability (a patient can carry several
     diagnoses at once -- multi-label). The training loop's BCE loss and the
     0.5 decision threshold both assume probabilities, not raw logits.
  3. Finiteness: no NaNs escape. The production sLSTM CUDA kernel runs in
     bfloat16 and can emit NaN on degenerate inputs -- which is why
     'train.py' wraps every batch in 'nan_to_num'. This test confirms the
     vanilla path is clean to begin with, so when a NaN does show up in a real
     run I know to blame the kernel or the data.

    pytest                                 # whole suite (this file included)
    pytest tests/test_model.py             # just this file
    pytest tests/test_model.py::test_model_output_shape   # one test

"""
import torch
import pytest  # noqa: F401  -- imported for parity with the rest of the suite;
                # no decorators are used in this module yet (see NOTE below).
from omegaconf import OmegaConf
# The single class under test. The import resolving at all is itself a weak
# smoke test: it confirms the third-party 'xlstm' package and the model module
# load cleanly under the "vanilla" backend on a CPU-only interpreter.
from src.models.xlstm_ecg import XLSTMECGModel


def get_test_config():
    """Build the minimal 'cfg.model' subtree the model needs, sized for speed.

    'XLSTMECGModel.__init__' reads only the 'cfg.model.*' keys.

    The values are the smallest that still keep the dimension contract intact,
    so the three tests build and run in well under a second on CPU.

    Returns
    -------
    omegaconf.DictConfig
        A config whose only populated subtree is 'model', ready to pass straight
        to 'XLSTMECGModel(cfg)'.
    """
    return OmegaConf.create({
        "model": {
            # Real feature width: 12 ECG leads x 241 STFT freq bins. Must match
            # the last dim of the random inputs in every test below.
            "input_size": 2892,
            "embedding_dim": 64,   # small for fast testing
            "num_blocks": 2,
            "num_heads": 4,
            "num_classes": 5,
            "dropout": 0.0,        # disable dropout for deterministic test
            "slstm_backend": "vanilla",
        }
    })


def test_model_output_shape():
    """Shape contract: '(B, T, input_size)' in -> '(B, num_classes)' out.
    """
    cfg = get_test_config()
    model = XLSTMECGModel(cfg)
    # (B=2, T=59, input_size=2892): two synthetic "ECGs" worth of STFT features.
    # B=2 (not 1) so a batch-axis bug can't slip through by collapsing to a scalar.
    x = torch.randn(2, 59, 2892)
    y = model(x)
    # Expect (B, num_classes) = (2, 5): the time axis is gone (pooled away) and
    # the feature axis has become the per-class probability vector.
    assert y.shape == (2, 5)


def test_model_output_range():
    """Range contract: every output is a probability in '[0, 1]'.
    The model ends in a sigmoid, so outputs have to be probabilities.
    """
    cfg = get_test_config()
    model = XLSTMECGModel(cfg)
    x = torch.randn(2, 59, 2892)
    y = model(x)
    # '.all()' reduces the (2, 5) boolean tensor to a single truth y value, so
    # the assertion holds only if every element is in range. In theory sigmoid
    # output is the open interval (0, 1); I assert the inclusive [0, 1] bound,
    # which also tolerates an exact 0.0 or 1.0.
    assert (y >= 0).all() and (y <= 1).all()


def test_model_no_nan():
    """Finiteness contract: the forward pass produces no NaNs on clean input.

    On random, well-conditioned input the vanilla backend should never emit NaN.
    (The production CUDA kernel runs in bfloat16 and can, which is why the
    training loop wraps every batch in 'nan_to_num'. This test isolates the
    model graph itself, so a NaN seen in a real run points to the kernel or the
    data.)
    """
    cfg = get_test_config()
    model = XLSTMECGModel(cfg)
    x = torch.randn(2, 59, 2892)
    y = model(x)
    # 'torch.isnan(y)' -> (2, 5) bool mask of NaN positions; '.any()' is True if
    # even one element is NaN. The assertion fails the moment a single NaN
    # appears anywhere in the output.
    assert not torch.isnan(y).any()
