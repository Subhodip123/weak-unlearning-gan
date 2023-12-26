import numpy as np

def sweep_config():
    """gives the confugaration for wandb sweep param tuning"""
    config = {
        'method': 'random',
        'name': 'base classifier hyperparam tune',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_loss'
            },
        'parameters': {
            'epochs': {'values':[100,200,300,400,500,600,700,800,900,1000]},
            'early_stop': {'values': [20,40,60,80,100]},
            'beta1': {'distribution':'uniform', 'max': .9, 'min': .7},
            'beta2':{'distribution':'uniform', 'max': 0.99, 'min': 0.8},
            'lr': {'distribution':'uniform', 'max': 0.1, 'min': 0.0001}
         }
    }
    return config

def config():
    """Gives the config dic for wandb run"""
    config_dic = {
        'epochs': 200,
        'early_stop': 100,
        'beta1':0.708216557158532,
        'beta2': 0.9720846792754038,
        'lr': 0.07874247637556907
    }
    return config_dic

def augconfig():
    """Gives the config dic for wandb run"""
    config_dic = {
        'epochs': 300,
        'early_stop': 80,
        'beta1': 0.8208655501906537,
        'beta2': 0.9815076608017378,
        'lr': 0.06383674873250486
    }
    return config_dic
