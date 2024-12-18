import torch
import pickle
import pandas as pd
from rdkit import Chem
from typing import Tuple, List

from coati.models.simple_coati2.io import load_coati2
from coati.common.util import batch_indexable
from coati.models.simple_coati2.transformer_only import COATI_Smiles_Inference
from coati.models.simple_coati2.trie_tokenizer import TrieTokenizer

from linear_regression_suite.admet_dataset import load_dataset, dataset_info

def load_model(model_path: str = "s3://terray-public/models/coati2_chiral_03-08-24.pkl") -> Tuple[COATI_Smiles_Inference, TrieTokenizer]:
    # Load the COATI model
    encoder, tokenizer = load_coati2(
        freeze=True,
        device=torch.device("cuda:0"),
        doc_url=model_path,
    )
    return encoder, tokenizer

def _embed_smiles_batch(smiles_list: List[str], encoder: COATI_Smiles_Inference, tokenizer: TrieTokenizer) -> torch.Tensor:
    tokenized_list = []

    for s in smiles_list:
      try:
        smiles = tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
      except:
        smiles = tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
      
      tokenized_list.append(smiles)
    
    batch_tokens = torch.tensor(
        tokenized_list,
        device=encoder.device,
        dtype=torch.int,
    )

    batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
    
    return batch_embeds

def embed_smiles_in_batches(smiles_list, encoder, tokenizer):
    # Convert to canonical SMILES and embed in batches
    smiles_batch = [Chem.MolToSmiles(Chem.MolFromSmiles(smile), canonical=True) if smile else None for smile in smiles_list]
    embeddings = []
    
    for i, batch in enumerate(batch_indexable(smiles_batch, batch_size=256)):
        print(f"Embedding batch {i+1}/{len(smiles_batch)//256}")
        batch_embeddings = _embed_smiles_batch(batch, encoder, tokenizer)
        embeddings.extend(batch_embeddings.cpu().numpy())

    return embeddings

def update_dataset_with_embeddings(file_path, encoder, tokenizer):
    data = load_dataset(file_path)
    
    if 'smiles' in data.columns:
        embeddings = embed_smiles_in_batches(data['smiles'], encoder, tokenizer)
        data['coati2'] = embeddings
    
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def update_loop(base_path: str='datasets/'):

    encoder, tokenizer = load_model()

    for dataset_name, (_, _) in dataset_info.items():
        print(f"Adding Embeddings to {dataset_name} dataset")
        file_path = f'{base_path}{dataset_name}.pkl'
        update_dataset_with_embeddings(file_path, encoder, tokenizer)
        print(f"Added Embeddings to {dataset_name} dataset\n\n")

if __name__ == "__main__":
    update_loop()