"""
src/models/xlstm_ecg.py
=======================
The xLSTM-ECG network for multi-label ECG classification, my re-implementation
of the original study (Kang et al., 2025; arXiv:2504.16101).

"""

import torch
import torch.nn as nn
from omegaconf import DictConfig  # type of the cfg passed to __init__
# The xlstm package (Beck/Hochreiter et al.) supplies the building blocks.
# I import the config dataclasses plus the 'xLSTMBlockStack' container, then
# deliberately reach past the container to its '.blocks' ModuleList so I can
# drive the fusion myself (see __init__ for why).
from xlstm import (
    xLSTMBlockStack,         # container that builds an ordered list of blocks
    xLSTMBlockStackConfig,   # top-level config: how many blocks, which type, etc.
    mLSTMBlockConfig,        # wraps one mLSTM layer (+ optional feed-forward)
    mLSTMLayerConfig,        # the matrix-LSTM layer hyperparameters
    sLSTMBlockConfig,        # wraps one sLSTM layer (+ feed-forward)
    sLSTMLayerConfig,        # the scalar-LSTM layer hyperparameters (CUDA kernel)
    FeedForwardConfig,       # the per-block GELU feed-forward sub-layer
)
# Use the library's own LayerNorm so my post-fusion norm matches the
# normalization used inside the blocks (same epsilon / parameterization).
from xlstm.components.ln import LayerNorm


class AttentionPooling(nn.Module):
    """
    Additive attention pooling over the time dimension.

    Given (B, T, D), this learns a per-time-step scalar score, softmax-
    normalizes it across T, and returns the score-weighted sum: (B, D). It is a
    learned weighted average over time -- the weights come from the data itself
    rather than being fixed, as they are in mean pooling. Clinically, this lets
    the head lean on the beats or frequency moments it finds most informative
    instead of treating every time step equally.

    Parameters
    ----------
    dim : int
        The embedding dimension D of the incoming sequence. The single linear
        scorer maps D -> 1; the only learned parameters are this small head
        (D weights + 1 bias), which is why the attention variant costs almost
        nothing in parameter count.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse the time axis of x into a single attention-weighted vector.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, D): a batch of length-T sequences of D-dim embeddings.

        Returns
        -------
        torch.Tensor of shape (B, D): the per-sequence attention-weighted sum.
        """
        # self.attn(x) scores every time step -> (B, T, 1); softmax over dim=1
        # (the T axis) makes the T weights a prob dist that sums to 1.
        w = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)

        # Broadcast-multiply the weights back onto the embeddings and sum over
        # T. The (B, T, 1) weights broadcast across the D channels of x, so the
        # weighted sum collapses T to one vector per (B, D).
        return (x * w).sum(dim=1)                 # (B, D)


class XLSTMECGModel(nn.Module):
    """
    xLSTM-ECG: multi-label ECG classification via feature fusion with xLSTM.
    Re-implements the original study (Kang et al., 2025; arXiv:2504.16101).

    Input:  (batch, seq_len, input_size) = (batch, 59, 2892) for the original
            study's nfft=480 / 12-lead default. seq_len is the STFT time dim T;
            input_size is 12*F where F = nfft//2 + 1.
    Output: (batch, num_classes) sigmoid probabilities in [0, 1].

    """

    def __init__(self, cfg: DictConfig):
        """Build the full model from a config snapshot.


        Parameters
        ----------
        cfg : omegaconf.DictConfig
            Only the 'cfg.model' subtree is read. The keys consumed are listed
            in the class docstring under 'cfg.model keys consumed'. Optional
            keys ('fusion_type', 'pooling') are read with 'getattr' defaults so
            older config / checkpoint snapshots still load.

        """
        super().__init__()

        # -- 1. Config snapshot -- store the few values I re-use in forward ---
        # Copy these scalars onto self rather than keeping a reference to cfg,
        # so the model carries no dependency on the (mutable) config object
        # after construction -- everything forward() needs is captured here.
        self.embedding_dim = cfg.model.embedding_dim
        self.num_classes   = cfg.model.num_classes
        self.num_blocks    = cfg.model.num_blocks

        self.fusion_type = getattr(cfg.model, "fusion_type", "layer")
        assert self.fusion_type in ("layer", "sequential",
                                     "slstm_only", "mlstm_only"), (
            f"Unknown fusion_type: {self.fusion_type!r}. "
            "Must be one of: layer | sequential | slstm_only | mlstm_only"
        )

        # -- 2. Input projection ----------------------------------------------
        # Maps (B, T, input_size) -> (B, T, embedding_dim). input_size for the
        # default config is 12 leads x 241 freq bins = 2892. With data.nfft
        # != 480 the caller (train.py) overrides this to 12 x (nfft//2 + 1).
        self.input_projection = nn.Linear(
            cfg.model.input_size, cfg.model.embedding_dim
        )

        self.dropout = nn.Dropout(p=cfg.model.dropout)
        slstm_backend = cfg.model.slstm_backend

        # -- 3a. sLSTM block stack (parallel branch #1) -----------------------
        # The xlstm library expects a stack config listing which positions get
        # sLSTM vs mLSTM blocks. I want a pure-sLSTM stack of depth num_blocks,
        # so slstm_at = [0, 1, ..., num_blocks-1].

        slstm_stack_cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=slstm_backend,
                    num_heads=cfg.model.num_heads,

                    # conv1d_kernel_size=4: a short causal depthwise conv on the
                    # gate pre-activations inside the sLSTM block. 4 is the
                    # xlstm default and the original study's setting; it gives a
                    # small local-context window before the recurrence.
                    conv1d_kernel_size=4,

                    # 'powerlaw_blockdependent' initializes the forget-gate bias
                    # so deeper blocks start biased toward longer memory -- a
                    # library-recommended init that stabilizes the sLSTM
                    # recurrence early in training.
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    # proj_factor=1.3: the feed-forward sub-layer expands the
                    # embedding by 1.3x internally (vs the usual 4x in a
                    # transformer MLP) -- the xlstm default, kept for fidelity.
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=59,           # STFT time dimension T
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
            # slstm_at lists which block positions are sLSTM. For a pure sLSTM
            # stack, every position 0..num_blocks-1 is listed.
            slstm_at=list(range(self.num_blocks)),    # all positions = sLSTM

            # because I add my OWN post-fusion LayerNorm
            # after the two stacks are combined (see self.post_blocks_norm below);
            # the library's trailing norm would otherwise double up.
            add_post_blocks_norm=False,
        )
        # NOTE: context_length is hardcoded to 59 in both stack configs. The
        # library uses it only to size internal buffers, and it must equal the
        # STFT time dimension T.

        # I pull '.blocks' (a ModuleList) instead of using the full
        # xLSTMBlockStack so I can drive fusion at each layer myself. The
        # container's own call would run every block end-to-end with no hook
        # for the per-block averaging that 'layer' fusion needs, so I discard
        # the wrapper and keep only its constructed ModuleList.
        self.slstm_blocks = xLSTMBlockStack(slstm_stack_cfg).blocks

        # -- 3b. mLSTM block stack (parallel branch #2) -----------------------
        # Symmetric to the sLSTM stack but with mLSTM blocks. slstm_at=[] tells
        # the library 'no sLSTM in this stack', so every position defaults to
        # mLSTM. Same kernel-size and head settings; mLSTM is the 'matrix LSTM'
        # with a different gating + attention pattern.
        mlstm_stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,            # same local conv as sLSTM
                    # qkv_proj_blocksize=4: the mLSTM forms query/key/value
                    # projections in blocks of this size (a library efficiency
                    # knob); 4 is the default and divides cleanly into the head
                    # dimension for the embedding_dims I use.
                    qkv_proj_blocksize=4,
                    num_heads=cfg.model.num_heads,
                ),

            ),
            context_length=59,                        # must match the sLSTM stack
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
            # slstm_at=[] -> NO position is sLSTM, so the library fills every
            # position with an mLSTM block. The mirror image of the
            # slstm_at=range(...) used above.
            slstm_at=[],                              # all positions = mLSTM
            add_post_blocks_norm=False,               # same reason as the sLSTM stack
        )
        self.mlstm_blocks = xLSTMBlockStack(mlstm_stack_cfg).blocks

        # -- 4. Post-fusion LayerNorm -----------------------------------------
        # Applied to the fused stack output before pooling. Stabilizes training
        # when the two stacks produce outputs at different scales.
        self.post_blocks_norm = LayerNorm(ndim=self.embedding_dim)

        # -- 5. Per-class binary classifiers ----------------------------------
        # Eq. 15 of the original study: one independent linear head per class,
        # mapping the pooled vector to a single logit. A ModuleList instead of
        # one big Linear(D, C) so each class can be trained independently
        # (relevant for the bce_weighted / focal loss variants I tried).
        # NOTE: numerically, C separate Linear(D, 1) heads are equivalent to a
        # single Linear(D, C); the per-head form is purely organizational -- it
        # makes per-class freezing and per-class weight inspection trivial.

        self.classifiers = nn.ModuleList([
            nn.Linear(self.embedding_dim, 1)
            for _ in range(self.num_classes)
        ])

        # Sigmoid turns the C concatenated logits into per-class probabilities.
        # Multi-label, NOT multi-class: each class gets its own sigmoid. A
        # softmax would force the probabilities to compete and sum to 1, which
        # is wrong clinically -- a single ECG can carry several diagnoses at
        # once (say, an old infarction plus a conduction block).
        self.sigmoid = nn.Sigmoid()

        # -- 6. Pooling strategy ----------------------------------------------
        # Pooling collapses the (B, T, D) sequence into (B, D). The original
        # study uses mean (eq. 14). I also support 'last' (single time step, an
        # ablation) and 'attention' (additive attention, within seed noise of mean).
        pool_type = getattr(cfg.model, "pooling", "mean")
        self.pool_type     = pool_type


        self.pooling_layer = (AttentionPooling(self.embedding_dim)
                              if pool_type == "attention" else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, input_size). For the original study's config:
            (B, 59, 2892), reshaped from (B, 12, 241, 59) by the train loop's
            input pipeline (see 'train.py:train_epoch').

        Returns
        -------
        torch.Tensor of shape (B, num_classes), values in [0, 1]. The caller
        reads these as independent per-class probabilities and thresholds at
        0.5 (or per-class tuned thresholds, see 'scripts/threshold_tune.py').
        """
        # -- Input projection + dropout --------------------------------------
        # The only place input_size is consumed; after this line every tensor in
        # the network has channel dim == embedding_dim.
        x = self.input_projection(x)
        x = self.dropout(x)

        # -- Feature fusion (per fusion_type) --------------------------------
        # Each branch below ends with 'x_fused' holding (B, T, embedding_dim);
        # the branches differ only in HOW the two stacks are combined.

        if self.fusion_type == "layer":
            # Layer fusion (eq. 13 of the original study): after each block,
            # both stacks receive the averaged output of the previous block.
            # This is iterative mutual refinement -- at each depth each stack
            # sees what the other made of the previous depth.
            x_fused = x
            # zip pairs the i-th sLSTM block with the i-th mLSTM block; the two
            # stacks are guaranteed equal length (both have num_blocks blocks).
            for slstm_block, mlstm_block in zip(self.slstm_blocks,
                                                  self.mlstm_blocks):
                # Both blocks see the SAME fused input from the previous depth.
                x_s     = slstm_block(x_fused)        # (B, T, embedding_dim)
                x_m     = mlstm_block(x_fused)        # (B, T, embedding_dim)
                # Average the two views and feed the result into the next depth.
                x_fused = (x_s + x_m) / 2.0           # arithmetic mean

        elif self.fusion_type == "sequential":
            # Sequential fusion (eq. 12 of the original study): each stack
            # processes the ORIGINAL input independently; outputs fused only at
            # the end. With num_blocks=1 this equals layer fusion. For
            # num_blocks >= 2, sequential gives each stack a clean view of the
            # input but loses the cross-stack interaction at intermediate
            # depths.
            # Run the sLSTM stack to completion on the input...
            x_s = x
            for slstm_block in self.slstm_blocks:
                x_s = slstm_block(x_s)
            # ...then the mLSTM stack, also starting from the original input
            # (NOT from x_s) -- this independence is the whole point of the
            # 'sequential' variant.
            x_m = x
            for mlstm_block in self.mlstm_blocks:
                x_m = mlstm_block(x_m)
            # Single fusion at the very end.
            x_fused = (x_s + x_m) / 2.0

        elif self.fusion_type == "slstm_only":
            # Ablation: sLSTM stack only (Table 5, row 1 of the original
            # study). The mLSTM blocks are still in the module tree (built
            # unconditionally) but never called -- they sit in eval mode with
            # their initial weights.
            x_fused = x
            for slstm_block in self.slstm_blocks:
                x_fused = slstm_block(x_fused)        # chain blocks in place

        else:  # mlstm_only
            # Ablation: mLSTM stack only (Table 5, row 2 of the original
            # study). Mirror of the slstm_only branch; here the sLSTM stack is
            # the unused one.
            x_fused = x
            for mlstm_block in self.mlstm_blocks:
                x_fused = mlstm_block(x_fused)

        # -- Post-fusion ------------------------------------------------------
        # LayerNorm stabilizes the stack output before pooling.
        x_fused = self.post_blocks_norm(x_fused)   # (B, T, embedding_dim)

        # Pool across time (eq. 14 of the original study). Every branch maps
        # the (B, T, D) sequence down to a single (B, D) vector 'v'.
        if self.pool_type == "mean":
            # Unweighted average over the T time steps -- the original study's
            # choice.
            v = x_fused.mean(dim=1)                # (B, embedding_dim)
        elif self.pool_type == "last":

            # Non-default ablation (pooling="last"): use only the final time step's
            # embedding as the sequence summary -- the classic "final RNN hidden state"
            # reduction (cf. Sutskever et al. 2014, seq2seq). The default/study choice
            # is mean pooling above.
            v = x_fused[:, -1, :]
        elif self.pool_type == "attention":
            # Learned weighted average (see AttentionPooling above).
            v = self.pooling_layer(x_fused)
        else:
            raise ValueError(f"Unknown pooling: {self.pool_type!r}")

        # Second dropout -- applied to the pooled vector before classification.
        # Reuses the same self.dropout module as after the input projection.
        v = self.dropout(v)

        # -- Per-class linear classifiers (eq. 15 of the original study) -----
        # Each Linear emits one logit of shape (B, 1); concatenating C of them
        # along dim=1 gives the (B, C) logit matrix.

        logits = torch.cat([clf(v) for clf in self.classifiers], dim=1)
        return self.sigmoid(logits)
