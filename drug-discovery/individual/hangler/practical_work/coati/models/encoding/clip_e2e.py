#
# This version does clip only on naked smiles strings.
# and does smiles => smiles e2e.
#

import random
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from torch.nn import functional as F

from coati.containers.rdkit_utils import disable_logger, permute_smiles
from coati.models.encoding.e3gnn_clip import e3gnn_clip
from coati.models.encoding.fill_in_middle import adj_mat_to_tokens
from coati.models.encoding.smiles_xformer import (
    RotarySmilesTransformer,
    SmilesTransformerConfig,
)
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer

disable_logger()


class clip_loss(nn.Module):
    """
    Directly out of the OpenAI CLIP paper
    """

    def __init__(self):
        super().__init__()

    def forward(self, smiles_features, conformer_features, bad_rows):
        logits_per_smiles = smiles_features @ conformer_features.T
        logits_per_conformer = conformer_features @ smiles_features.T
        num_logits = logits_per_smiles.shape[0]
        labels = torch.arange(
            num_logits, device=smiles_features.device, dtype=torch.long
        )
        labels = torch.where(bad_rows, -1 * torch.ones_like(labels), labels)
        total_loss = (
            F.cross_entropy(logits_per_smiles, labels, ignore_index=-1)
            + F.cross_entropy(logits_per_conformer, labels, ignore_index=-1)
        ) / 2
        return total_loss.unsqueeze(0)  # for dataparallel.


def clip_ar_xform(
    batch: Dict[str, Any],
    tokenizer: TrieTokenizer,
    p_dataset: float = 0.2,
    p_formula: float = 0.2,
    p_fim: float = 0.0,
    p_graph: float = 0.0,
    p_clip: float = 0.9,
    p_clip_cut: float = 0.3,
    p_randsmiles: float = 0.0,  # NOTE: this is applied BEFORE raw_tokens are collected if 0.
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
    coord_noise: bool = False,
):
    """
    This version randomly augments data in several ways (front or back)
    it randomly prepends the dataset, it randomly prepends the
    molecular formula and randomly performs the prefix, suffix, middle

    Note: this exploits the fact that the encoding will always begin with [SMILES]
    and end with [STOP] [PAD] [PAD] ... and that [PAD] is always token mapping to 0

    Here's examples.
    smiles: "c1ccccc1"
    No augmentations: [SMILES][c1][cccc][c1][STOP]

    dataset aug: [SET][tensormol][SMILES][c1][cccc][c1][STOP] or
                 [SMILES][c1][cccc][c1][SET][tensormol][STOP]
    formula aug: [FORMULA][ELM6][NUM6][ELM1][NUM6][SMILES][c1][cccc][c1][STOP] or
                 [SMILES][c1][cccc][c1][FORMULA][ELM6][NUM6][ELEMENT][ELM1][NUM6][STOP] or
    partialform: [ELM6][NUM6][SMILES][c1][cccc][c1][STOP]
    graph aug  : [GRAPH][NODE][ELM1][NUM1][ELM1][NUM2][EDGE1][NUM1][NUM2][BOND1]
    parital    : [NODE][ELM1][NUM1][NODE][ELM1][NUM2][EDGE1][NUM1][NUM2][BOND1]

    Fill-in-middle is always applied AFTER previous possible augmentations
    so dataset can be filled in if desired but not on the string level
    so these special tokens aren't broken.

    fim aug:
    [ELEMENT][Tk6][Tk6][ELEMENT][Tk1][Tk6][SMILES][c1][cccc][c1][STOP] becomes:
    [PREFIX][ELEMENT][Tk6][Tk6][ELEMENT][Tk1][Tk6][SMILES][c1][SUFFIX][c1][MIDDLE][cccc]
    """
    # print first entry in batch
    print("batch in clip_ar_xform:\n", batch)

    assert "smiles" in batch
    assert "source_collection" in batch
    assert "atoms" in batch
    assert "coords" in batch

    token_stack = []
    s2s_stack = []
    from rdkit import Chem

    for k, S__ in enumerate(batch["smiles"]):
        S_ = Chem.CanonSmiles(S__)
        try:
            reps = ["smiles"]
            # Choose augmentations
            if random.random() < p_dataset:
                if (
                    "[" + batch["source_collection"][k] + "]"
                    in tokenizer.special_tokens
                ):
                    reps.append("set")
            if random.random() < p_formula:
                reps.append("formula")
            if (
                random.random() < p_graph
                and "adj_mat" in batch
                and "adj_mat_atoms" in batch
            ):
                reps.append("graph")

            random.shuffle(reps)
            S = ""
            for rep in reps:
                if rep == "set":
                    S = S + "[SET]" + "[" + batch["source_collection"][k] + "]"
                elif rep == "smiles":
                    S = S + "[SMILES]" + S_
                elif rep == "formula":
                    ats = batch["atoms"][k].astype(int)
                    cts = np.bincount(ats[ats > 0])
                    if (cts < 150).all():
                        rows = np.stack(
                            [np.arange(0, cts.shape[0])[cts > 0], cts[cts > 0]], -1
                        )
                        formula_string = "[FORMULA]" + "".join(
                            [
                                "[ELM" + str(r[0]) + "][NUM" + str(r[1]) + "]"
                                for r in rows
                            ]
                        )
                    else:
                        formula_string = ""
                    S = S + formula_string
                elif rep == "graph":
                    graph_string = adj_mat_to_tokens(
                        batch["adj_mat"][k], batch["adj_mat_atoms"][k]
                    )
                    S = S + graph_string

            S = S + "[STOP]"
            ttext = tokenizer.tokenize_text(S, pad=False, range_check=False)

            if random.random() < p_clip and len(ttext) > 3:
                if random.random() < p_clip_cut:
                    # CLIP Augmentation (hidden token always goes to position [1])
                    # Pull out the stop token.
                    # try [CLIP][UNK][SMILES][SUFFIX][MIDDLE].....
                    stop_token = ttext.pop()
                    # choose the positions of [MIDDLE] and [SUFFIX]
                    # note: they cannot be the first two tokens.
                    mp, sp = 1, 1
                    while mp == sp:  #  or mp > ttext_ind_smiles:
                        mp, sp = sorted(
                            [
                                random.randint(2, len(ttext)),
                                random.randint(2, len(ttext)),
                            ]
                        )
                    ttext = (
                        tokenizer.tokenize_text(
                            "[CLIP][UNK]", pad=False, range_check=False
                        )
                        + ttext[:mp]
                        + tokenizer.tokenize_text(
                            "[SUFFIX]", pad=False, range_check=False
                        )
                        + ttext[sp:]
                        + tokenizer.tokenize_text(
                            "[MIDDLE]", pad=False, range_check=False
                        )
                        + ttext[mp:sp]
                        + [stop_token]
                    )
                else:
                    ttext = (
                        tokenizer.tokenize_text(
                            "[CLIP][UNK]", pad=False, range_check=False
                        )
                        + ttext
                    )
            elif random.random() < p_fim and len(ttext) > 4:
                # Fill-in-middle Augmentation
                # Pull out the stop token.
                stop_token = ttext.pop()
                # choose the positions of [MIDDLE] and [SUFFIX]
                # note: they cannot be the first token.
                mp, sp = 1, 1
                while mp == sp:
                    mp, sp = sorted(
                        [random.randint(1, len(ttext)), random.randint(1, len(ttext))]
                    )
                ttext = (
                    tokenizer.tokenize_text("[PREFIX]", pad=False, range_check=False)
                    + ttext[:mp]
                    + tokenizer.tokenize_text("[SUFFIX]", pad=False, range_check=False)
                    + ttext[sp:]
                    + tokenizer.tokenize_text("[MIDDLE]", pad=False, range_check=False)
                    + ttext[mp:sp]
                    + [stop_token]
                )

            if random.random() < p_randsmiles:
                S_raw = "[SMILES]" + permute_smiles(S_) + "[STOP]"
                s2s_text = tokenizer.tokenize_text(S_raw, pad=False, range_check=False)
                unnperm_toks = tokenizer.tokenize_text(
                    "[SMILES]" + S_ + "[STOP]", pad=False, range_check=False
                )
            else:
                S_raw = "[SMILES]" + S_ + "[STOP]"
                s2s_text = tokenizer.tokenize_text(S_raw, pad=False, range_check=False)
                unnperm_toks = s2s_text

            if len(ttext) <= tokenizer.n_seq and len(s2s_text) <= tokenizer.n_seq:
                t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                t[: len(ttext)] = torch.tensor(ttext)

                smi_t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                smi_t[: len(s2s_text)] = torch.tensor(s2s_text)

                token_stack.append(t)
                s2s_stack.append(smi_t)
            else:
                # try to just make it a simple smiles if it got oversized.
                # But still always canonically decode (token stack gets unpermed)
                if (
                    len(s2s_text) <= tokenizer.n_seq
                    and len(unnperm_toks) <= tokenizer.n_seq
                ):
                    t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                    t[: len(unnperm_toks)] = torch.tensor(unnperm_toks)

                    smi_t = torch.zeros(
                        tokenizer.n_seq, dtype=torch.long, device=device
                    )
                    smi_t[: len(s2s_text)] = torch.tensor(s2s_text)

                    token_stack.append(t)
                    s2s_stack.append(smi_t)
                else:
                    s2s_stack.append(
                        torch.cat(
                            [
                                tokenizer.stop_token
                                * torch.ones(1, dtype=torch.long, device=device),
                                torch.zeros(
                                    tokenizer.n_seq - 1, dtype=torch.long, device=device
                                ),
                            ],
                            0,
                        )
                    )
                    token_stack.append(
                        torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                    )
                    print("Too much seq data.", S_raw, len(s2s_text))
                    continue

        except Exception as Ex:
            print("Tokenize failure:", S_, " Except:", Ex)
            # raise Ex
            s2s_stack.append(
                torch.cat(
                    [
                        tokenizer.stop_token
                        * torch.ones(1, dtype=torch.long, device=device),
                        torch.zeros(
                            tokenizer.n_seq - 1, dtype=torch.long, device=device
                        ),
                    ],
                    0,
                )
            )
            token_stack.append(
                torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
            )
            continue

    batch["tokens"] = torch.stack(token_stack, 0)
    batch["raw_tokens"] = torch.stack(s2s_stack, 0)

    if batch["atoms"].shape[0] < 1:
        raise Exception("empty batch")

    for col in ["tokens", "atoms", "raw_tokens"]:
        if col in batch:
            if type(batch[col]) != torch.Tensor:
                batch[col] = torch.tensor(batch[col], requires_grad=False).to(
                    device, torch.long
                )
    for col in ["coords"]:
        if col in batch:
            if type(batch[col]) != torch.Tensor:
                batch[col] = torch.tensor(batch[col], requires_grad=False).to(
                    device, dtype
                )
    if coord_noise:
        batch["coords"] += torch.normal(
            torch.zeros_like(batch["coords"]), 0.05 * torch.ones_like(batch["coords"])
        )

    # decrease the sequence size to max demanded by this batch.
    batch["tokens"] = batch["tokens"][:, : (batch["tokens"].sum(0) > 0).sum()]
    batch["raw_tokens"] = batch["raw_tokens"][
        :, : (batch["raw_tokens"].sum(0) > 0).sum()
    ]

    # Alignment in cross entropy:
    # [SMILES],  [TOKEN 1]... [STOP] [PAD]
    # [TOKEN 1] [TOKEN 2]... [PAD]  [PAD]
    batch["y_next"] = torch.zeros_like(batch["tokens"])
    batch["y_next"][:, : (batch["tokens"].shape[1] - 1)] = batch["tokens"][
        :, 1:
    ].clone()
    # Critical! no loss for predictions of the pad
    batch["y_next"][batch["y_next"] == tokenizer.clip_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.pad_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.unk_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.suffix_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.middle_token] = -1
    return batch


def _tokenize_smiles(
    smi, tokenizer, prefix="[SMILES]", suffix="[STOP]", device="cpu", max_size=None
):
    if max_size is None:
        max_size = tokenizer.n_seq
    try:
        ttext = tokenizer.tokenize_text(
            prefix + smi + suffix, pad=False, range_check=False
        )
        if len(ttext) <= max_size:
            t = torch.zeros(max_size, dtype=torch.long, device=device)
            t[: len(ttext)] = torch.tensor(ttext)
            return t
    except KeyError:
        pass


class e3gnn_smiles_clip_e2e(nn.Module):
    """
    Adds routines for the AR-Generation.
    and a forward pass which only requires
    one pass through each encoder type.
    """

    def __init__(
        self,
        n_layer_e3gnn: int = 4,
        n_layer_xformer: int = 16,
        n_hidden_xformer: int = 128,
        n_hidden_e3nn: int = 128,
        msg_cutoff_e3nn: float = 4.0,
        n_embd_common: int = 128,
        n_head: int = 8,
        n_seq: int = 200,
        n_tok: int = 4,
        biases: bool = True,
        torch_emb: bool = False,
        residual: bool = False,
        norm_clips: bool = True,
        norm_embed: bool = False,
        token_mlp: bool = True,  # Do we use a nonlinear MLP to convert HCLIP into a token.
        use_point_encoder: bool = True,  # if false, do not use a point encoder at all.
        old_architecture: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.embed_dim = n_embd_common
        self.point_encoder = e3gnn_clip(
            device=device,
            dtype=dtype,
            hidden_nf=n_hidden_e3nn,
            message_cutoff=msg_cutoff_e3nn,
            dropout=0.0,
            torch_emb=torch_emb,
            residual=residual,
            n_layers=n_layer_e3gnn,
        )
        kwargs = {
            "n_layer": n_layer_xformer,
            "n_embd": n_hidden_xformer,
            "n_head": n_head,
            "n_seq": n_seq,
            "n_tok": n_tok,
            "device": device,
            "dtype": dtype,
            "biases": biases,
            "norm_embed": norm_embed,
        }
        self.xformer_config = SmilesTransformerConfig(**kwargs)
        self.xformer = RotarySmilesTransformer(self.xformer_config)
        self.device = device
        self.use_point_encoder = use_point_encoder
        # Each of these get a linear mapping into the common hidden space.

        if norm_clips:
            if old_architecture:
                self.point_to_clip = nn.Sequential(
                    nn.Linear(self.point_encoder.hidden_nf, self.embed_dim),
                    nn.LayerNorm(self.point_encoder.hidden_nf),
                )
                self.smiles_to_clip = nn.Sequential(
                    nn.Linear(self.xformer.n_embd, self.embed_dim),
                    nn.LayerNorm(self.embed_dim),
                )
            else:
                self.point_to_clip = nn.Sequential(
                    nn.LayerNorm(self.point_encoder.hidden_nf),
                    nn.Linear(self.point_encoder.hidden_nf, self.embed_dim),
                )
                self.smiles_to_clip = nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.xformer.n_embd, self.embed_dim),
                )
        else:
            self.point_to_clip = nn.Linear(self.point_encoder.hidden_nf, self.embed_dim)
            self.smiles_to_clip = nn.Linear(self.xformer.n_embd, self.embed_dim)

        if token_mlp:
            # A mapping to make the special token(s?).
            self.point_clip_to_special_tokens = nn.Sequential(
                nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim)
            )
        else:
            self.point_clip_to_special_tokens = nn.Identity()

        n_params_e3gnn = sum(p.numel() for p in self.point_encoder.parameters())
        n_params_smiles = sum(p.numel() for p in self.xformer.parameters())
        n_params = n_params_e3gnn + n_params_smiles
        print(
            f"number of parameters Total: {n_params_e3gnn/1e6:.2f}M xformer: {n_params_smiles/1e6:.2f}M Total: {n_params/1e6:.2f}M "
        )
        self.clip_loss = clip_loss()
        self.to(self.device)

    def encode_tokens(self, token_indices: torch.Tensor, tokenizer) -> torch.Tensor:
        """
        Embeds the tokens, and projects into the latent space.
        """
        return self.smiles_to_clip(self.xformer.encode(token_indices, tokenizer))

    def encode_points(self, atoms: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Embeds coordinates/atoms and projects into the latent space.

        If point encoder is not used, this returns a zero tensor.
        """
        if self.use_point_encoder:
            return self.point_to_clip(self.point_encoder(atoms, coords))
        else:
            return torch.zeros(atoms.shape[0], self.embed_dim).to(self.device)

    def points_to_2d(
        self,
        atoms: torch.Tensor,
        coords: torch.Tensor,
        tokenizer: TrieTokenizer,
        fill_in_from: str = "[SMILES]",
        noise_scale: float = 0.0,  # add noise to the point encoding.
        inv_temp: float = 2,  # inverse temperature for sampling.
        k: int = 100,  # top k sampling.
    ):
        """
        generation of SMILES (or GRAPH)
        from atoms and coords
        """
        assert fill_in_from == "[SMILES]" or fill_in_from == "[GRAPH]"
        h_clip = self.encode_points(atoms, coords)
        if noise_scale > 0:
            h_clip += torch.normal(
                mean=torch.zeros_like(h_clip), std=noise_scale * torch.ones_like(h_clip)
            )
        h_token = self.point_clip_to_special_tokens(h_clip)
        # create a 'batch' to infer smiles.
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + "[SUFFIX][MIDDLE]", pad=False
        )
        generation = self.xformer.generate_topk_with_inj(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token[0],
        )
        if fill_in_from == "[SMILES]":
            return tokenizer.decode(generation, special=False)
        else:
            return tokenizer.decode(generation)

    def hclip_to_2d(
        self,
        h_clip: torch.Tensor,
        tokenizer: TrieTokenizer,
        fill_in_from: str = "[SMILES]",
        noise_scale: float = 0.0,
        do_suffix: bool = False,
        inv_temp: float = 2,
        k: int = 100,
    ):
        """
        Testing generation of SMILES (or GRAPH)
        from atoms and coords
        """
        assert fill_in_from == "[SMILES]" or fill_in_from == "[GRAPH]"
        if noise_scale > 0:
            h_clip += torch.normal(
                mean=torch.zeros_like(h_clip), std=noise_scale * torch.ones_like(h_clip)
            )
        h_token = self.point_clip_to_special_tokens(h_clip)
        # create a 'batch' to infer smiles.
        if do_suffix:
            suffstr = "[SUFFIX][MIDDLE]"
        else:
            suffstr = ""
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + suffstr, pad=False
        )
        generation = self.xformer.generate_topk_with_inj(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token[0],
        )
        if fill_in_from == "[SMILES]":
            return tokenizer.decode(generation, special=False)
        else:
            return tokenizer.decode(generation)

    def hclip_to_2d_batch(
        self,
        h_clip: torch.Tensor,
        tokenizer: TrieTokenizer,
        fill_in_from: str = "[SMILES]",
        noise_scale: float = 0.0,
        inv_temp: float = 2,
        k: int = 100,
        do_suffix=False,
        keep_special: bool = False,
        return_tokens: bool = False,
    ):
        """
        Decodes a batch of h_clip embeddings into a batch of SMILES (or GRAPH)
        """
        if noise_scale > 0:
            h_clip += torch.normal(
                mean=torch.zeros_like(h_clip), std=noise_scale * torch.ones_like(h_clip)
            )
        h_token = self.point_clip_to_special_tokens(h_clip)
        # Add Stefan
        print("h_token", h_token.shape)
        print("h_token", h_token)
        
        if do_suffix:
            suffstr = "[SUFFIX][MIDDLE]"
        else:
            suffstr = ""
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + suffstr, pad=False
        )
        generation = self.xformer.generate_top_k_with_inj_batch(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            pad_token=tokenizer.pad_token,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token,
        )
        smiles_list = [
            tokenizer.decode(token_out, special=keep_special)
            for token_out in generation
        ]

        if return_tokens:
            return smiles_list, generation

        return smiles_list

    def points_to_2d_batch(
        self,
        atom_batch: torch.Tensor,
        coords_batch: torch.Tensor,
        tokenizer: TrieTokenizer,
        fill_in_from: str = "[SMILES]",
        noise_scale: float = 0.0,
        do_suffix: bool = False,
        inv_temp: float = 2,
        k=100,
        keep_special=False,
    ):
        """
        generation of SMILES (or GRAPH)
        from atoms and coords
        """
        h_clip = self.encode_points(atom_batch, coords_batch)
        if noise_scale > 0:
            h_clip += torch.normal(
                mean=torch.zeros_like(h_clip), std=noise_scale * torch.ones_like(h_clip)
            )
        h_token = self.point_clip_to_special_tokens(h_clip)
        if do_suffix:
            suffstr = "[SUFFIX][MIDDLE]"
        else:
            suffstr = ""
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + suffstr, pad=False
        )
        generation = self.xformer.generate_top_k_with_inj_batch(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            pad_token=tokenizer.pad_token,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token,
        )
        smiles_list = [
            tokenizer.decode(token_out, special=keep_special)
            for token_out in generation
        ]
        return smiles_list

    def hclip_and_tokens_to_likelihood(
        self, hclip: torch.Tensor, smiles: str, tokenizer: TrieTokenizer
    ) -> torch.Tensor:
        """
        Simply computes the likelihood that hclip decodes to a given smiles.
        """
        tokens = torch.tensor(
            tokenizer.tokenize_text(
                "[CLIP][UNK][SMILES][SUFFIX][MIDDLE]" + smiles + "[STOP]", pad=False
            ),
            device=hclip.device,
            dtype=torch.long,
        ).unsqueeze(0)
        y_next = torch.zeros_like(tokens)
        y_next[:, : (tokens.shape[1] - 1)] = tokens[:, 1:].clone()
        y_next[y_next == tokenizer.clip_token] = -1
        y_next[y_next == tokenizer.pad_token] = -1
        y_next[y_next == tokenizer.smiles_token] = -1
        y_next[y_next == tokenizer.unk_token] = -1
        y_next[y_next == tokenizer.suffix_token] = -1
        y_next[y_next == tokenizer.middle_token] = -1
        logits = self.xformer.forward_with_replacement(
            tokens, self.point_clip_to_special_tokens(hclip.unsqueeze(0)), tokenizer
        )
        ar_loss_ = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_next.view(-1),
            ignore_index=-1,
            reduction="none",
        ).reshape(tokens.shape)
        ar_loss_[y_next == -1] = 0
        return ar_loss_.sum(-1)

    def batch_smiles_to_s2s_likelihood(
        self, smiles: List[str], tokenizer: TrieTokenizer
    ) -> torch.Tensor:
        """Simply computes the likelihood that SMILES->hclip->SMILES decodes for all SMILES in a list of `smiles`"""
        # make tokens from '<smi>[STOP]'
        _tokens = [
            _tokenize_smiles(
                smi,
                tokenizer,
                prefix="",
                suffix="[STOP]",
                device=self.device,
                max_size=(tokenizer.n_seq - 5),
            )
            for smi in smiles
        ]
        tokenizes_mask = torch.tensor(
            [False if t is None else True for t in _tokens],
            dtype=torch.bool,
            device=self.device,
        )
        _tokens = torch.stack([t for t in _tokens if t is not None]).to(self.device)

        # make embeddings from '[SMILES]<smi>[STOP]'
        hclip_tokens = torch.zeros(
            _tokens.shape[0],
            _tokens.shape[1] + 1,  # leave space for [SMILES]
            dtype=torch.long,
            device=self.device,
        )
        hclip_tokens[:, 0] = tokenizer.smiles_token
        hclip_tokens[:, 1:] = _tokens
        hclip = self.encode_tokens(hclip_tokens, tokenizer)

        # make logits from '[CLIP][UNK][SMILES][SUFFIX][MIDDLE]<smi>[STOP]' and hclip from [SMILES]<smi>[STOP]
        tokens = torch.zeros(
            _tokens.shape[0],
            _tokens.shape[1] + 5,  # leave space for [CLIP][UNK][SMILES][SUFFIX][MIDDLE]
            dtype=torch.long,
            device=self.device,
        )
        tokens[:, 0] = tokenizer.clip_token
        tokens[:, 1] = tokenizer.unk_token
        tokens[:, 2] = tokenizer.smiles_token
        tokens[:, 3] = tokenizer.suffix_token
        tokens[:, 4] = tokenizer.middle_token
        tokens[:, 5:] = _tokens

        logits = self.xformer.forward_with_replacement(  # pred token
            tokens, self.point_clip_to_special_tokens(hclip), tokenizer
        )

        # calculate cross entropy on pred token and actual next token
        mask_val = -1  # to mask loss for special tokens
        next_tokens = torch.zeros_like(tokens)
        next_tokens[:, : (tokens.shape[1] - 1)] = tokens[:, 1:].clone()
        next_tokens[
            :, :4
        ] = mask_val  # '[UNK][SMILES][SUFFIX][MIDDLE]', [CLIP] not present because it was first in tokens
        next_tokens[
            :, -1
        ] = mask_val  # because of the shift and next_tokens construction, next_tokens[:, -1] is [0, 0, ...], ensure this is pad token
        next_tokens[next_tokens == tokenizer.pad_token] = mask_val

        # find ar_loss per SMILES
        ar_loss_ = (
            torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[2]),
                next_tokens.view(-1),
                ignore_index=mask_val,
                reduction="none",
            )
            .view(next_tokens.shape[0], next_tokens.shape[1])
            .sum(axis=1)
        )
        return ar_loss_, tokenizes_mask

    def complete_batch(
        self,
        prefixes: List[str],
        tokenizer: TrieTokenizer,
        inv_temp: float = 2,
        k: int = 100,
        keep_special: bool = False,
        de_fim: bool = True,
    ):
        """
        Testing generation of SMILES
        from atoms and coords
        """
        # create a 'batch' to infer smiles.
        tokens = [tokenizer.tokenize_text(p, pad=False) for p in prefixes]
        generation = self.xformer.generate_topk_batch(
            prefix=tokens,
            stop_token=tokenizer.stop_token,
            pad_token=tokenizer.pad_token,
            inv_temp=inv_temp,
            k=k,
        )
        smiles_list = [
            tokenizer.decode(token_out, special=keep_special, de_fim=de_fim)
            for token_out in generation
        ]
        return smiles_list

    def forward_dist(
        self,
        raw_tokens: torch.Tensor,
        augmented_tokens: torch.Tensor,
        atoms: torch.Tensor,
        coords: torch.Tensor,
        tokenizer,
        p_clip_emb_smi: float = 0.4,
    ):
        """
        Same as the below routine but for DistributedDataParallel training.
        """
        with autocast(enabled=False, device_type="cuda"):
            h_e3gnn = self.encode_points(atoms, coords)
            h_smiles = self.encode_tokens(raw_tokens, tokenizer)
            try:
                assert h_e3gnn.shape[0] == h_smiles.shape[0]
            except Exception as Ex:
                print(
                    Ex,
                    raw_tokens.shape,
                    augmented_tokens.shape,
                    atoms.shape,
                    coords.shape,
                    h_e3gnn.shape,
                    h_smiles.shape,
                )
                raise Ex
            point_clip_token = self.point_clip_to_special_tokens(h_e3gnn)
            smiles_clip_token = self.point_clip_to_special_tokens(h_smiles)
            clip_token = torch.where(
                (torch.rand((h_e3gnn.shape[0],), device=atoms.device) > p_clip_emb_smi)
                .unsqueeze(-1)
                .repeat(1, point_clip_token.shape[-1]),
                point_clip_token,
                smiles_clip_token,
            )

        logits = self.xformer.forward_with_replacement(
            augmented_tokens, clip_token, tokenizer
        )
        bad_rows = augmented_tokens.sum(-1) < 1
        return h_e3gnn, h_smiles, logits, bad_rows

    def forward(
        self,
        raw_tokens: torch.Tensor,
        augmented_tokens: torch.Tensor,
        atoms: torch.Tensor,
        coords: torch.Tensor,
        tokenizer,
        p_clip_emb_smi: float = 0.4,
    ):
        """
        Same as the below routine but for DistributedDataParallel training.
        """
        with autocast(enabled=False, device_type="cuda"):
            h_e3gnn = self.encode_points(atoms, coords)
            h_smiles = self.encode_tokens(raw_tokens, tokenizer)
            assert h_e3gnn.shape[0] == h_smiles.shape[0]
            point_clip_token = self.point_clip_to_special_tokens(h_e3gnn)
            smiles_clip_token = self.point_clip_to_special_tokens(h_smiles)
            clip_token = torch.where(
                (torch.rand((h_e3gnn.shape[0],), device=atoms.device) > p_clip_emb_smi)
                .unsqueeze(-1)
                .repeat(1, point_clip_token.shape[-1]),
                point_clip_token,
                smiles_clip_token,
            )
        logits = self.xformer.forward_with_replacement(
            augmented_tokens, clip_token, tokenizer
        )
        bad_rows = augmented_tokens.sum(-1) < 1
        return h_e3gnn, h_smiles, logits, self.clip_loss(h_smiles, h_e3gnn, bad_rows)
