#### Model weights for TUSZ data

18 channels in the following order: \
`['FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']`

To access and use the `.pth` file, do: 
>weights = torch.load(weights_path) \
model = CLEEGN(n_chan=18, fs=128.0, N_F=18).to(device) \
model.load_state_dict(weights["state_dict"])

(see also `inference.py`)

*Attention:* 
- Only trained on a small part of TUSZ dataset (10 patients, 10min each)
- Yet **no normalization** used for training (tbd)
