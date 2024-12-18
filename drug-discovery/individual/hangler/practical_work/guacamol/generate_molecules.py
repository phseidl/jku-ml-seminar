import torch
from abc import ABC, abstractmethod
from typing import list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MoleculeGenerator(ABC):
    """
    Abstract base class for molecule generators.
    """
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Method to load the model.
        """
        pass

    @abstractmethod
    def generate_molecules(self, num_molecules: int):
        """
        Method to generate molecules.
        """
        pass

    def validate_molecules(self, molecules: list) -> int:
        """
        Validate generated molecules.
        """
        from rdkit import Chem
        return sum(1 for mol in molecules if Chem.MolFromSmiles(mol) is not None)

    def save_molecules(self, molecules: list, filename: str = "generated_molecules.csv"):
        """
        Save generated molecules to a file.
        """
        with open(filename, 'w') as f:
            for mol in molecules:
                f.write(mol + '\n')

        print(f"Saved generated molecules to {filename}")


class COATIGenerator(MoleculeGenerator):
    def __init__(self, device: torch.device, doc_url: str):
        super().__init__(device)
        self.doc_url = doc_url
        self.model, self.tokenizer = self.load_model()

    def load_model(self, freeze: bool = True, print_debug: bool = True):
        from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
        # Assuming `load_e3gnn_smiles_clip_e2e` returns a model and tokenizer
        return load_e3gnn_smiles_clip_e2e(freeze=freeze, device=self.device, doc_url=self.doc_url, print_debug=print_debug)

    def generate_molecules(self, num_molecules: int = 100, prefix: str = "[SMILES]", k: int = 100, inv_temp: float = 2.0):
        # Example of generating molecules
        prefixes = [prefix for _ in range(num_molecules)]
        generated_mols = []

        for i in range(num_molecules//100):
            mols = self.model.complete_batch(
                prefixes=prefixes, 
                tokenizer=self.tokenizer, 
                k=k, inv_
                temp=inv_temp
            )

            generated_mols.extend(mols)

        return generated_mols


class COATI2Generator(MoleculeGenerator):
    def __init__(self, device: torch.device, doc_url: str):
        super().__init__(device)
        self.doc_url = doc_url
        self.model, self.tokenizer = self.load_model()

    def load_model(self, freeze: bool = True):
        from coati.models.simple_coati2.io import load_coati2
        # Assuming `load_coati2` returns a model and tokenizer
        return load_coati2(freeze=freeze, device=self.device, doc_url=self.doc_url)

    def generate_molecules(self, num_molecules: int = 100, h_coati = torch.randn(100, 512), prefix: str = "[SET][SMILES]", k: int = 100, inv_temp: float = 2.0):
        generated_mols = []

        for i in range(num_molecules//100):
            mols = self.model.hcoati_to_2d_batch(
                h_coati = h_coati.to(self.device),
                tokenizer=self.tokenizer,
                fill_in_from=prefix,
                k=k,
                inv_temp=inv_temp
            )
            generated_mols.extend(mols)

        return generated_mols

if __name__ == "__main__":
    # Instantiate and use the COATI generator
    # Guacamol Model
    print("Start generating Molecules with Guacamol Model")

    guacamol_coati = COATIGenerator(device, "./guacamol/e3gnn_smiles_clip_e2e_1685977071_1686087379.pkl")
    generated_molecules = guacamol_coati.generate_molecules(10000)
    valid_count = guacamol_coati.validate_molecules(generated_molecules)
    guacamol_coati.save_molecules(generated_molecules, "guacamol_generated_molecules.csv")

    print(f"Generated {len(generated_molecules)} molecules")
    print(f"Valid molecules: {valid_count}")

    # Grande Closed Model
    print("Start generating Molecules with Grande Closed Model")

    grande_closed_coati = COATIGenerator(device, "s3://terray-public/models/grande_closed.pkl")
    generated_molecules = grande_closed_coati.generate_molecules(10000, prefix="[SET][SMILES]")
    valid_count = grande_closed_coati.validate_molecules(generated_molecules)
    grande_closed_coati.save_molecules(generated_molecules, "grande_closed_generated_molecules.csv")

    print(f"Generated {len(generated_molecules)} molecules")
    print(f"Valid molecules: {valid_count}")

    # Autoregressive Only Model
    print("Start generating Molecules with Autoregressive Only Model")

    autoreg_coati = COATIGenerator(device, "s3://terray-public/models/autoreg_only.pkl")
    generated_molecules = autoreg_coati.generate_molecules(10000)
    valid_count = autoreg_coati.validate_molecules(generated_molecules)
    autoreg_coati.save_molecules(generated_molecules, "autoreg_only_generated_molecules.csv")
    
    print(f"Generated {len(generated_molecules)} molecules")
    print(f"Valid molecules: {valid_count}")

    # COATI2 Model
    print("Start generating Molecules with COATI2 Model")

    coati2 = COATI2Generator(device, "s3://terray-public/models/coati2_chiral_03-08-24.pkl")
    generated_molecules = coati2.generate_molecules(10000, prefix="[SET][SMILES]")
    valid_count = coati2.validate_molecules(generated_molecules)
    coati2.save_molecules(generated_molecules, "coati2_generated_molecules.csv")

    print(f"Generated {len(generated_molecules)} molecules")
    print(f"Valid molecules: {valid_count}")