import os
# Set OpenMP environment variable to handle runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Set the path to your project directory
project_path = Path("/Users/stefanhangler/Documents/Uni/Msc_AI/Thesis/Code.nosync/jku-ml-seminar23/drug-discovery/individual/hangler/thesis")
sys.path.append(str(project_path))

import torch
from transformers import BertTokenizer
from train_autoencoder import ldmol_autoencoder
from utils import AE_SMILES_encoder
from rdkit import Chem

class SMILESEncoder:
    def __init__(self, checkpoint_path, tokenizer_path, config_path):
        # Load tokenizer
        self.tokenizer = BertTokenizer(vocab_file=tokenizer_path, do_lower_case=False, do_basic_tokenize=False)
        # Load autoencoder model
        self.ae_config = {
            'bert_config_decoder': str(config_path / 'config_decoder.json'),
            'bert_config_encoder': str(config_path / 'config_encoder.json'),
            'embed_dim': 256,
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ldmol_autoencoder(config=self.ae_config, no_train=True, tokenizer=self.tokenizer).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
    
    def encode_smiles(self, smiles_list):
        valid_smiles = []
        for smile in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    valid_smiles.append(canonical_smile)
            except:
                print(f"Warning: Could not process SMILES: {smile}")
                continue
        
        if not valid_smiles:
            raise ValueError("No valid SMILES strings to process")
            
        embeddings = AE_SMILES_encoder(valid_smiles, self.model).permute(0, 2, 1).unsqueeze(-1)
        return embeddings
    

if __name__ == "__main__":
    try:
        project_path = Path("/Users/stefanhangler/Documents/Uni/Msc_AI/Thesis/Code.nosync/jku-ml-seminar23/drug-discovery/individual/hangler/thesis")
        sys.path.append(str(project_path))
        
        smiles_path = project_path / "data/pubchem_100smiles_test.txt"
        model_path = project_path / "Pretrained/checkpoint_autoencoder.ckpt"
        tokenizer_path = project_path / "vocab_bpe_300_sc.txt"
        config_path = project_path
        
        encoder = SMILESEncoder(model_path, tokenizer_path, config_path)
        
        with open(smiles_path, 'r') as f:
            smiles_list = [line.strip() for line in f.readlines()]
        
        embeddings = encoder.encode_smiles(smiles_list)
        print(f"Embeddings shape: {embeddings.shape}")
        
        torch.save(embeddings, project_path / "smiles_embeddings.pt")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")