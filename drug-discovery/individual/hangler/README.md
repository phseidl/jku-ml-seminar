# Project Status

🚧 **In Progress**

This project is currently under active development:

- **Codebase Changes**: The code is evolving, which means directory paths to data or imports may not be accurate at the moment. We're in the process of standardizing these paths for ease of use.
- **Initialization**: The initial setup for this project is not yet fully automated. **To get started, you'll need to manually download the COATI-Repository and store the notebooks in it before executing them.** The plan is to automate this in the near future!

## Getting Started

For now, please follow these steps to set up the project:

1. Clone or download the [COATI-Repository](https://github.com/terraytherapeutics/COATI) to your local machine.
2. Place the current Code and Notebooks in the local COATI-Repository to use its functions.

## Project Progress

### Done

- Look at Tokenization &rarr; `Cl` has its own token because the Trie-encoder uses the max. possible number of characters for one Token.
- Successfully extracted embeddings from Coati. Embeddings with associated SMILES have been saved as a `.csv` file.
- Downloaded the Baseline Guacamol dataset for further experimentation and comparison.

### In Process

- Preprocessing the Guacamol dataset to add 3D coordinates and atoms to the corresponding SMILES strings.
- Training COATI on the Guacamol train and valitdation set to evaluate performance.

### Next Steps

- Compare the results with established benchmarks such as [Guacamol](https://github.com/BenevolentAI/guacamol), [MolGPT](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600), and [MolReactGen](https://github.com/hogru/MolReactGen) to assess COATI's performance.
- Implement linear probing on molecule net as done in CLAMP.
- Utilize the embeddings for a classification task.

