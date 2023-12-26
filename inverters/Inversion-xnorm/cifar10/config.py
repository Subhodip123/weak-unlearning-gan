import numpy as np

def sweep_config():
    """gives the confugaration for wandb sweep param tuning"""
    config = {
        'method': 'random',
        'name': 'inverter paramtune',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_loss'
            },
        'parameters': {
            'batch_size': {'values': [64, 128, 256]},
            'last_layer': {'values': [64, 128, 256, 512]},
            'early_stop':{'values': [20,40,60,80,100]},
            'epochs': {'values': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
            'beta1': {'distribution':'uniform', 'max': .9, 'min': .8},
            'beta2':{'distribution':'uniform', 'max': 0.99, 'min': 0.9},
            'lr': {'distribution':'uniform', 'max': 0.1, 'min': 0.0001}
         }
    }
    return config

def config():
    """Gives the config dic for wandb run"""
    config_dic = {
        'batch_size': 512,
        'last_layer': 1024,
        'early_stop':60,
        'epochs': 400,
        'beta1':0.9,
        'beta2': 0.9,
        'lr': 0.01443984513454316
        
    }
    return config_dic
