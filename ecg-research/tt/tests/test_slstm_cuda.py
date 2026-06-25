"""
tests/test_slstm_cuda.py
========================
GPU smoke test for the production sLSTM CUDA backend used by real training runs.

How it is run
-------------
    pytest                                   # whole suite; this module skips off-GPU
    pytest tests/test_slstm_cuda.py          # just this file
    pytest tests/test_slstm_cuda.py::test_slstm_cuda_forward_backward
    pytest -rs tests/test_slstm_cuda.py      # -rs shows the skip reason on a CPU box


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
    """Compile the sLSTM CUDA kernel and run one forward + backward pass.

    Fabricates a tiny random batch and builds its
    own block-stack config, and it runs only when the module-level skip
    condition (GPU present and ninja on PATH) is satisfied. It returns nothing;
    failure is signaled by an assertion or by an exception escaping the
    forward/backward call, per pytest convention.
    """
    # Build a minimal pure-sLSTM block stack that matches the production config
    # in src/models/xlstm_ecg.py closely enough to compile the same kernel
    # variant. The kernel is keyed by (embedding_dim, num_heads), so those two
    # values decide which compiled artifact gets used or built.
    cfg = xLSTMBlockStackConfig(
        slstm_block=sLSTMBlockConfig(

            # backend='cuda' is the whole reason this test exists -- it selects
            # the JIT-compiled kernel rather than the 'vanilla' PyTorch path the
            # rest of the suite uses. num_heads=4 / conv1d_kernel_size=4 /
            # bias_init='powerlaw_blockdependent' are copied verbatim from the
            # production sLSTM layer config so I exercise the real code path, not
            # a toy one.
            slstm=sLSTMLayerConfig(backend="cuda", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"),

            # proj_factor=1.3 + gelu mirror the production feed-forward sub-layer
            # (xlstm library default; see src/models/xlstm_ecg.py). They don't
            # affect whether the kernel compiles, but keeping them identical
            # makes this stack structurally the same block the model builds.
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        # context_length=59 is the STFT time dimension T used everywhere in the
        # pipeline (12 leads x 241 freq bins reshaped to seq_len=59). The library
        # uses it to size internal buffers, so the seq dim of x below must match.
        # T is the number of STFT time frames per beat-window the model reads.
        context_length=59,

        # Two blocks (rather than one) so the stack is genuinely deep -- this
        # exercises the kernel being invoked at more than one position and, with
        # bias_init='powerlaw_blockdependent', a block-dependent forget-gate init.
        num_blocks=2,

        # embedding_dim=256 is the kernel's hidden width. Together with num_heads
        # it fixes the per-head dim (256/4 = 64). Kept modest so the one-time
        # compile and the forward/backward stay fast on any GPU.
        embedding_dim=256,

        # slstm_at lists which positions in the stack are sLSTM blocks. [0, 1]
        # makes BOTH positions sLSTM -> a pure-sLSTM stack (no mLSTM here), which
        # is what isolates the sLSTM kernel under test.
        # NOTE: this is [0, 1] spelled out rather than list(range(num_blocks)) as
        # the model does; fine for a fixed 2-block test, but it has to stay in
        # sync with num_blocks above if that count ever changes.
        slstm_at=[0, 1],

        # add_post_blocks_norm=False matches production (the model applies its
        # own post-fusion LayerNorm); here it just means the stack returns the
        # raw block output, so the shape assertion below is exact.
        add_post_blocks_norm=False,
    )
    # Move the whole stack onto the GPU. The first .to('cuda') call on a fresh
    # (embedding_dim, num_heads) combo is what triggers the ninja JIT compile;
    # if ninja or nvcc were misconfigured this line is where it would blow up.
    stack = xLSTMBlockStack(cfg).to("cuda")

    # Synthetic input shaped (batch=4, seq_len=59, embedding_dim=256). seq_len
    # must equal context_length and the last dim must equal embedding_dim, or the
    # kernel's internal reshape would mismatch. Random data is sufficient for a
    # shape/gradient smoke test -- we are not checking learned behavior.
    x = torch.randn(4, 59, 256, device="cuda")  # small batch first

    # Forward pass through the compiled kernel. The sLSTM block stack is shape-
    # preserving (a sequence of embeddings in, a sequence of the same size out),
    # so the output must have exactly the same shape as the input.
    y = stack(x)

    # (1) Shape contract: a healthy stack returns (4, 59, 256). A mismatch here
    # almost always means a head/stride bug in the kernel or a misconfigured dim.
    assert y.shape == x.shape

    # Collapse the output to a single scalar so there is something to backprop
    # from. .mean() is an arbitrary, cheap reduction -- the value is irrelevant;
    # all I need is a valid scalar loss to drive autograd.
    loss = y.mean()

    # (2) Backward contract: this exercises the kernel's custom backward. If it
    # raises (unimplemented grad, NaN guard, bad context save), the test fails.
    # I don't inspect the gradients themselves -- reaching this line without an
    # exception is the assertion.
    loss.backward()
