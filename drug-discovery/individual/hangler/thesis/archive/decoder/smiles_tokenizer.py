import re
from rdkit import Chem

# Define the regex pattern for tokenizing SMILES
_ELEMENTS_STR = (
    r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|"
    r"Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
)
PATTERN = rf"(\[[^\]]+\]|{_ELEMENTS_STR}|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%\d{{2}}|\d)"

# Compile the regex once for efficiency
token_regex = re.compile(PATTERN)

def segment_smiles(smiles: str):
    """
    Tokenize a SMILES string into its constituent tokens.
    Args:
        smiles (str): A SMILES string.
    Returns:
        list: A list of tokens from the SMILES string.
    """
    return token_regex.findall(smiles)

def validate_smiles(smiles: str):
    """
    Validate SMILES strings using RDKit.
    Returns True if valid, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None