import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.distributions.categorical import Categorical

from decoder.decoder_xbert import XbertDecoder
from decoder.vocab import Vocabulary

#####################################
# 1) SINGLE-STEP SAMPLING FUNCTION
#####################################
@torch.no_grad()
def generate_one_step(
    model: XbertDecoder,
    context_embeddings: torch.Tensor,
    partial_seq: torch.Tensor,
    k: int = None,
    stochastic: bool = True
):
    """
    Generate the next token for each sequence in the batch (single decoding step).

    Args:
        model: Your XbertDecoder (LightningModule) instance.
        context_embeddings: shape [batch_size, embed_dim].
        partial_seq: shape [batch_size, cur_seq_len] of token IDs (already generated).
        k: If not None, top-k sampling or top-k greedy is used. If None, normal sampling/greedy with no top-k.
        stochastic: If True => sampling, If False => greedy.

    Returns:
        next_tokens: shape [batch_size, 1] or (log_probs, topk_indices) if k is set.
    """
    # partial_seq => feed into model(...), retrieve the logits for the last position
    batch_size, cur_len = partial_seq.shape
    attention_mask = (partial_seq != model.pad_idx).long()
    
    # forward pass: shape [batch_size, cur_len, vocab_size]
    logits = model(context_embeddings, partial_seq, attention_mask=attention_mask)
    # last step's logits: shape [batch_size, vocab_size]
    last_logits = logits[:, -1, :]

    # top-k path
    if k is not None and k > 1:
        probs = torch.softmax(last_logits, dim=-1)
        if stochastic:
            # top-k sampling
            topk_tokens = torch.multinomial(probs, num_samples=k, replacement=False)  # [batch_size, k]
            # gather log probabilities
            log_probs = torch.log(
                torch.stack([probs[i][topk_tokens[i]] for i in range(batch_size)], dim=0)
            )  # shape [batch_size, k]
            return log_probs, topk_tokens
        else:
            # top-k greedy
            topk_vals, topk_idx = torch.topk(last_logits, k, dim=-1)  # each: [batch_size, k]
            # convert to log_probs
            log_probs = torch.log_softmax(topk_vals, dim=-1)  # [batch_size, k]
            return log_probs, topk_idx

    # if k == 1 or k is None => normal single token (sampling or greedy)
    if stochastic:
        # sampling
        probs = torch.softmax(last_logits, dim=-1)
        next_token = Categorical(probs).sample().unsqueeze(1)  # [batch_size, 1]
    else:
        # greedy
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # [batch_size, 1]

    return next_token


#####################################
# 2) FULL-SEQUENCE DECODING FUNCTION
#####################################
@torch.no_grad()
def decode_smiles_batch(
    model: XbertDecoder,
    vocab: Vocabulary,
    embeddings: torch.Tensor,
    stochastic: bool = False,
    k: int = 1,
    max_length: int = 150,
):
    """
    Decode a batch of embeddings into SMILES strings using top-k or greedy sampling.

    Args:
        model: XbertDecoder.
        vocab: Vocabulary (with <sos>, <eos>, <pad>, etc.).
        embeddings: shape [batch_size, embed_dim].
        stochastic: If True, do sampling; if False, do greedy decoding.
        k: If k>1 => top-k decoding; if k=1 => normal decode (either sampling or greedy).
        max_length: Max tokens to generate (stops earlier if <eos> is generated).

    Returns:
        A list of strings (SMILES) of length batch_size.
    """
    model.eval()
    device = embeddings.device

    # If your model uses a linear layer to adapt embeddings, do it:
    if model.encode_prefix is not None:
        embeddings = model.encode_prefix(embeddings)  # shape [batch_size, hidden_size]

    batch_size = embeddings.size(0)
    start_token_id = vocab.start_idx
    end_token_id = vocab.end_idx
    pad_token_id = vocab.pad_idx

    # Initialize the partial sequences with <sos>
    partial_seqs = torch.full(
        (batch_size, 1),
        fill_value=start_token_id,
        dtype=torch.long,
        device=device
    )  # shape [batch_size, 1]

    # We'll store the final decoded sequences as lists of token IDs
    if k == 1:
        # Single-sequence decoding (greedy or sampling)
        for _ in range(max_length):
            # next_token => shape [batch_size, 1]
            next_token = generate_one_step(
                model=model,
                context_embeddings=embeddings,
                partial_seq=partial_seqs,
                k=None,                  # no top-k
                stochastic=stochastic
            )
            # If everything is <pad>, we can break early (rare)
            if next_token.sum() == 0:
                break
            # Append to partial seq
            partial_seqs = torch.cat([partial_seqs, next_token], dim=-1)
        
        # Convert each row to a SMILES string
        decoded_smiles = []
        for row in partial_seqs:
            tokens = row.tolist()
            # Truncate at <eos>
            if end_token_id in tokens:
                eos_pos = tokens.index(end_token_id)
                tokens = tokens[:eos_pos]
            # Remove the <sos> at the beginning
            if len(tokens) > 0 and tokens[0] == start_token_id:
                tokens = tokens[1:]
            # Convert to actual tokens -> string
            token_strs = vocab.decode(tokens)  # returns a list of tokens
            decoded_smiles.append("".join(token_strs))
        return decoded_smiles

    else:
        # top-k approach
        decoded_smiles = []
        for i in range(batch_size):
            # handle one embedding at a time
            single_emb = embeddings[i].unsqueeze(0)  # [1, hidden_size]
            single_seq = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)
            current_scores, current_seq_candidates = None, None
            final_outputs = []

            # initial top-k
            values, indices = generate_one_step(
                model=model,
                context_embeddings=single_emb,
                partial_seq=single_seq,
                k=k,
                stochastic=stochastic
            )  # each shape [1, k]
            # shape => values: [1, k], indices: [1, k]
            # expand single_seq into k candidate sequences
            seq_candidates = []
            for idx in range(k):
                seq_candidates.append(
                    torch.cat([single_seq, indices[:, idx:idx+1]], dim=-1)
                )  # shape [1, 2]
            seq_candidates = torch.cat(seq_candidates, dim=0)  # shape [k, 2]

            # current_scores => shape [k]
            current_scores = values[0]  # shape [k]

            # iterate up to max_length
            for step in range(max_length):
                # generate top-k for each existing candidate
                next_scores_list, next_candidates_list = [], []
                # shape of seq_candidates => [k, cur_len]
                for c_idx in range(seq_candidates.size(0)):
                    cand_seq = seq_candidates[c_idx].unsqueeze(0)  # [1, cur_len]
                    # one-step decode => returns log_probs [1, k], tokens [1, k]
                    step_values, step_indices = generate_one_step(
                        model=model,
                        context_embeddings=single_emb,
                        partial_seq=cand_seq,
                        k=k,
                        stochastic=stochastic
                    )
                    # shape => step_values: [1, k], step_indices: [1, k]
                    # we add them to current_scores[c_idx]
                    for s_idx in range(k):
                        new_score = current_scores[c_idx] + step_values[0, s_idx]
                        new_seq = torch.cat([cand_seq, step_indices[:, s_idx:s_idx+1]], dim=-1)
                        # Check if <eos> => store final output
                        if new_seq[0, -1].item() == end_token_id:
                            final_outputs.append((new_score.item(), new_seq[0]))
                        else:
                            next_scores_list.append(new_score)
                            next_candidates_list.append(new_seq[0])

                # If we accumulate enough final outputs, we can break early
                if len(final_outputs) >= k:
                    break

                if len(next_scores_list) == 0:
                    # No more expansions (all ended with <eos>), break
                    break

                # Now pick the top-k from all expansions
                next_scores_tensor = torch.stack(next_scores_list, dim=0)  # shape [X]
                # get the top k
                topk_vals, topk_idxs = torch.topk(next_scores_tensor, k)
                # reorder next_candidates_list
                new_seq_candidates = []
                for order_idx in topk_idxs.tolist():
                    new_seq_candidates.append(next_candidates_list[order_idx].unsqueeze(0))
                seq_candidates = torch.cat(new_seq_candidates, dim=0)
                current_scores = topk_vals

            # pick the best final output (highest score)
            if len(final_outputs) > 0:
                final_outputs.sort(key=lambda x: x[0], reverse=True)
                best_seq = final_outputs[0][1]  # [seq_len]
            else:
                # if no final outputs ended with <eos>, just pick from seq_candidates
                best_idx = torch.argmax(current_scores)
                best_seq = seq_candidates[best_idx]

            # remove <sos> and <eos>
            token_ids = best_seq.tolist()
            if token_ids[0] == start_token_id:
                token_ids = token_ids[1:]
            if end_token_id in token_ids:
                eos_pos = token_ids.index(end_token_id)
                token_ids = token_ids[:eos_pos]

            tokens_str = vocab.decode(token_ids)
            decoded_smiles.append("".join(tokens_str))

        return decoded_smiles


###############################################
# 3) METRICS ON STRING MATCHES (OPTIONAL)
###############################################
def compute_string_metrics(
    ground_truth_smiles: List[str],
    predicted_smiles: List[str]
) -> Tuple[float, float, float, float]:
    """
    String-level exact match metrics (accuracy, precision, recall, f1).
    For single-label exact matching, precision = recall = accuracy => f1 = accuracy.

    If you want chemically-correct metrics, consider canonicalizing or
    checking molecular equivalence with RDKit.
    """
    if len(ground_truth_smiles) != len(predicted_smiles):
        raise ValueError("Mismatch in length of ground_truth_smiles vs. predicted_smiles.")

    total = len(ground_truth_smiles)
    correct = sum(gt == pred for gt, pred in zip(ground_truth_smiles, predicted_smiles))

    accuracy = correct / total if total > 0 else 0.0
    precision = accuracy
    recall = accuracy
    f1 = accuracy
    return accuracy, precision, recall, f1


###############################################
# 4) CHUNKED DECODING + METRICS (FOR LARGE DATA)
###############################################
def decode_and_evaluate(
    checkpoint_path: str,
    config_path: str,
    vocab: Vocabulary,
    embeddings_npy: str,
    smiles_parquet: str,
    k: int = 1,
    stochastic: bool = False,
    batch_size: int = 256,
    max_length: int = 150,
    device: str = "cuda",
    n_eval: int = -1
):
    """
    1) Loads your XbertDecoder checkpoint.
    2) Loads embeddings from a .npy file.
    3) Loads corresponding SMILES from parquet.
    4) Decodes in batches using `decode_smiles_batch(...)`.
    5) Computes string-level metrics.

    Args:
        checkpoint_path: Path to the .ckpt file saved by PyTorch Lightning.
        config_path: Path to the BERT config .json used to initialize XbertDecoder.
        vocab: Your Vocabulary instance (<sos>, <eos>, etc.).
        embeddings_npy: Path to the .npy file with shape [N, embed_dim].
        smiles_parquet: Path to a parquet with at least N rows (column 'smiles').
        k: top-k sampling or beam size. k=1 => pure greedy or sampling.
        stochastic: If True => sampling, if False => greedy.
        batch_size: Batch size for decoding.
        max_length: Max decoding steps (tokens).
        device: 'cuda' or 'cpu'.
        n_eval: If >0, decode only the first n_eval embeddings + SMILES.

    Returns:
        predicted_smiles: list of decoded strings.
        metrics: (accuracy, precision, recall, f1).
    """
    # 1) Load the checkpoint
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 128,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_decoder': './config_decoder_clamp.json',
        'schedular': {'sched': 'cosine', 'lr': 0.5e-4, 'epochs': 5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 0.5e-4, 'weight_decay': 0.02}
    }

    model = XbertDecoder(
        config=config,
        pad_idx=vocab.pad_idx,
        embed_dim=768,  # adapt if different
        use_linear=True
    )
    model.load_state_dict(ckpt_data["state_dict"], strict=True)
    model.to(device)
    model.eval()

    # 2) Load embeddings
    all_embs = np.load(embeddings_npy)  # shape [N, embed_dim]
    if n_eval > 0:
        all_embs = all_embs[:n_eval]
    all_embs_t = torch.tensor(all_embs, dtype=torch.float32, device=device)

    # 3) Load SMILES parquet
    df_smiles = pd.read_parquet(smiles_parquet)
    if len(df_smiles) < len(all_embs):
        raise ValueError("Not enough rows in SMILES parquet vs. embeddings.")
    ground_truth = df_smiles["smiles"].tolist()
    if n_eval > 0:
        ground_truth = ground_truth[:n_eval]

    # 4) Decode in batches
    total_size = len(all_embs_t)
    num_batches = (total_size + batch_size - 1) // batch_size
    predictions = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, total_size)
        batch_emb = all_embs_t[start:end]

        batch_decoded = decode_smiles_batch(
            model=model,
            vocab=vocab,
            embeddings=batch_emb,
            stochastic=stochastic,
            k=k,
            max_length=max_length
        )
        predictions.extend(batch_decoded)

    # 5) Compute metrics
    acc, prec, rec, f1 = compute_string_metrics(ground_truth, predictions)
    print(f"Decoded {len(predictions)} molecules with k={k}, stochastic={stochastic}")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1        = {f1:.4f}")

    return predictions, (acc, prec, rec, f1)