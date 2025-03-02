from models.OneD_ResCNN import OneD_ResCNN
from models.IC_U_NET import IC_U_NET, ICUNet_Git
from models.cleegn import CLEEGN

def model_select(model_class, model_cfg, data_cfg, window_size, downsample=False):
    if model_class == 'CLEEGN':
        model = CLEEGN(n_chan=model_cfg['n_chan'], fs=data_cfg['sfreq'], N_F=model_cfg['N_F'])
    elif model_class == 'IC_U_Net':
        model = IC_U_NET(input_channels=model_cfg['n_chan'])
    elif model_class == 'OneD_Res_CNN':
        if downsample:
            model = OneD_ResCNN(seq_length=int(data_cfg['sfreq']*window_size/2), batch_size=model_cfg['batch_size'], n_chan=model_cfg['n_chan'])
        else:
            model = OneD_ResCNN(seq_length=int(data_cfg['sfreq']*window_size), batch_size=model_cfg['batch_size'], n_chan=model_cfg['n_chan'])
    return model