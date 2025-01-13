import torch
import numpy as np
import pandas as pd
from decoder.vocab import Vocabulary
from decoder.decoder_inference_sampling import decode_and_evaluate, decode_smiles_batch
from decoder.decoder_xbert import XbertDecoder


if __name__ == "__main__":
    # 1) Load your Vocabulary
    vocab_file = "decoder/vocab.txt"
    with open(vocab_file, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    vocab = Vocabulary(tokens)

    # 2) Paths for full dataset test
    checkpoint_path = "checkpoints/decoder-epoch=00-val_loss=0.0000.ckpt"
    config_path = "config_decoder.json"
    embeddings_npy = "data/single embeddings/embeddings_pubchem10m.npy_0.npy"
    smiles_parquet = "data/pubchem_100smiles_test.txt"

    # 3) Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ##############################################
    # Test decoding for the first 5 SMILES
    ##############################################
    print("Running a short test on the first 5 SMILES from the dataset...")

    # Load all embeddings and SMILES
    all_embs = np.load(embeddings_npy)  # shape [N, 768]
    #df_smiles = pd.read(smiles_parquet)

    # load .txt file
    with open(smiles_parquet, "r") as f:
        smiles = f.readlines()
    smiles = [s.strip() for s in smiles]
    df_smiles = pd.DataFrame(smiles, columns=["CanonicalSMILES"])

    # Take the first 5 embeddings and SMILES
    test_embs = all_embs[:5]
    print(df_smiles[:5])
    test_smiles = df_smiles["CanonicalSMILES"][:5].tolist()

    # Convert to torch tensor
    test_embeddings_tensor = torch.tensor(test_embs, dtype=torch.float32, device=device)

    # Load checkpoint and model
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    model = XbertDecoder(
        config_path=config_path,
        pad_idx=vocab.pad_idx,
        embed_dim=768,  # Must match embedding dimension
        use_linear=True
    )
    model.load_state_dict(ckpt_data["state_dict"], strict=True)
    model.to(device)
    model.eval()

    # Decode 5 SMILES (greedy, k=1)
    decoded_smiles = decode_smiles_batch(
        model=model,
        vocab=vocab,
        embeddings=test_embeddings_tensor,
        stochastic=False,  # False => greedy
        k=1,
        max_length=150
    )

    print("\nDecoded vs. Ground-Truth (first 5 molecules):")
    for idx, (pred, gt) in enumerate(zip(decoded_smiles, test_smiles)):
        is_same = ("MATCH" if pred == gt else "DIFF")
        print(f"{idx+1}) GT:  {gt}")
        print(f"   Pred: {pred}")
        print(f"   --> {is_same}\n")

    ##############################################
    # Full dataset evaluation
    ##############################################
    # print("Running full dataset evaluation...")
    # predictions, metrics = decode_and_evaluate(
    #     checkpoint_path=checkpoint_path,
    #     config_path=config_path,
    #     vocab=vocab,
    #     embeddings_npy=embeddings_npy,
    #     smiles_parquet=smiles_parquet,
    #     k=1,
    #     stochastic=False,  # Greedy decoding
    #     batch_size=256,
    #     max_length=150,
    #     device=device,
    #     n_eval=-1  # Decode all embeddings in the dataset
    # )

    # # Print evaluation metrics
    # print("Full dataset evaluation metrics:")
    # print(f"  Accuracy:  {metrics[0]:.4f}")
    # print(f"  Precision: {metrics[1]:.4f}")
    # print(f"  Recall:    {metrics[2]:.4f}")
    # print(f"  F1:        {metrics[3]:.4f}")
