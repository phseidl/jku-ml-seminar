# Zero-Shot Conditional Molecule Generation with Latent Diffusion Models from Contrastive Pre-Trained Embeddings

*The code is based on the following paper:*
LDMol: Text-Conditioned Molecule Diffusion Model Leveraging Chemically Informative Latent Space. ([arxiv 2024](https://arxiv.org/abs/2405.17829))

*Architecture and Pipeline of the original paper for better understanding:*
![ldmol_fig2](https://github.com/jinhojsk515/LDMol/assets/59189526/1a172fed-39ab-44a6-848b-1740c7b37df4)

Original model checkpoint and data from the LDMol paper can be found [here](https://drive.google.com/drive/folders/170znWA5u3nC7S1mzF7RPNP5faAn56Q45?usp=sharing).


## Requirements
Run `conda env create -f requirements.yaml` and it will generate a conda environment.


## Acknowledgement
* The code for the latent Diffusion part is based on & modified from the official code of [LDMol](https://github.com/jinhojsk515/ldmol).
* The code and models for the encoder part are based on & modified from the official code of [CLAMP](https://github.com/ml-jku/clamp).