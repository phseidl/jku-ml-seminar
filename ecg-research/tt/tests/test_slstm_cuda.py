"""
tests/test_slstm_cuda.py
========================
GPU smoke test for the production sLSTM CUDA backend used by real training runs.

How it is run
-------------
    pytest                                   # whole suite; this module skips off-GPU
    pytest tests/test_slstm_cuda.py          # just this file


"""
import shutil

import pytest
import torch
# Build the sLSTM block stack from the library's own primitives rather than
# going through the project's XLSTMECGModel. The point of this test is to light
# up the CUDA kernel directly, so I want the thinnest possible wrapper around it
# -- no model-specific reshaping, fusion, or classifier head in the way.
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

# The sLSTM 'cuda' backend JIT-compiles a CUDA kernel, which needs both a GPU
# and the 'ninja' build tool on PATH. Skip the whole module when either is
# missing so a plain 'pytest' run does not error at collection on a CPU box.
# The CPU-only tests in test_data.py / test_model.py use slstm_backend='vanilla'
# and always run.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("ninja") is None,
    reason="sLSTM cuda backend requires a GPU and the ninja build tool",
)


def test_slstm_cuda_forward_backward():
    """Compile the sLSTM CUDA kernel and run one forward + backward pass on fake data.
    """
    # Build a minimal pure-sLSTM block stack that matches the production config
    # in src/models/xlstm_ecg.py closely enough to compile the same kernel
    # variant.
    cfg = xLSTMBlockStackConfig(
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(backend="cuda", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),

        context_length=59,

        # Two blocks (rather than one) so the stack is genuinely deep -- this
        # exercises the kernel being invoked at more than one position and, with
        # bias_init='powerlaw_blockdependent', a block-dependent forget-gate init.
        num_blocks=2,

        embedding_dim=256,

        # slstm_at lists which positions in the stack are sLSTM blocks. [0, 1]
        # makes BOTH positions sLSTM -> a pure-sLSTM stack (no mLSTM here), which
        # is what isolates the sLSTM kernel under test.
        # NOTE: this is [0, 1] spelled out rather than list(range(num_blocks)) as
        # the model does; fine for a fixed 2-block test, but it has to stay in
        # sync with num_blocks above if that count ever changes.
        slstm_at=[0, 1],

        add_post_blocks_norm=False,
    )
    # Move the whole stack onto the GPU. The first .to('cuda') call on a fresh
    # (embedding_dim, num_heads) combo is what triggers the ninja JIT compile;
    # if ninja or nvcc were misconfigured this line is where it would blow up.
    stack = xLSTMBlockStack(cfg).to("cuda")

    # Synthetic input (batch=4, seq_len=59, embedding_dim=256). seq_len must
    # equal context_length and the last dim embedding_dim, or the kernel's
    # reshape would mismatch; random data is fine for a shape/gradient smoke test.
    x = torch.randn(4, 59, 256, device="cuda")

    y = stack(x)

    assert y.shape == x.shape

    # Collapse the output to a single scalar so there is something to backprop
    # from.
    loss = y.mean()
    loss.backward()
