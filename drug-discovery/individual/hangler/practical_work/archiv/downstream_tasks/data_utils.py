import pandas as pd
from rdkit import Chem

def get_largest_component(smiles: str) -> str:
    """Extract the largest component from a SMILES string."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    
    # if '*' is present in the mol string replace it with ''
    if '*' in smiles:
        smiles = smiles.replace('*', '')
        mol = Chem.MolFromSmiles(smiles)
        
    components = Chem.GetMolFrags(mol, asMols=True)
    largest_mol = max(components, key=lambda component: component.GetNumAtoms())

    return Chem.MolToSmiles(largest_mol)

def prepare_data(smiles_data_path: str, activity_data_path: str) -> pd.DataFrame:
    """Load and merge SMILES and activity data into a single DataFrame."""
    
    print(f"Loading SMILES data from {smiles_data_path}")
    
    smis_df = pd.read_parquet(smiles_data_path)
    act_df = pd.read_parquet(activity_data_path)

    smis_df['CanonicalSMILES'] = smis_df['CanonicalSMILES'].apply(get_largest_component)

    combined_df = pd.merge(smis_df, act_df, left_on='CID', right_on='compound_idx')

    return combined_df
