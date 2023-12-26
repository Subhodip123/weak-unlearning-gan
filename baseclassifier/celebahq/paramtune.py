import os
import torch
from torch.optim import Adam, Adamax
import numpy as np
from datagen import Data
import wandb
from model import BaseClassifier
from train import training
from config import sweep_config
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if os.path.exists('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results') == False:
    os.makedirs('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results')

pos_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/celebahq"
neg_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_neg_imgs/celebahq"


wandb.login()


def paramtune(config=None):
    """Trains the classifier"""
    length = 50
    with wandb.init(config=config):
       config = wandb.config
       data = Data(pos_path=pos_path, neg_path=neg_path, length=length)
       train_dataloader, val_dataloader, train_labelloader, val_labelloader = data.dataloader()
       model = BaseClassifier(inchannels=3)
       model = model.to(device=device)
       optimizer = Adamax([param for param in model.parameters() if param.requires_grad == True ], lr = config.lr, betas=(config.beta1, config.beta2))
       no_epochs = int(config.epochs)
       training(model=model, model_optim=optimizer, no_epochs=no_epochs, train_dataloader=train_dataloader,
                train_labelloader=train_labelloader, val_dataloader=val_dataloader,
                val_labelloader=val_labelloader, length=length, early_stop=config.early_stop)


if __name__=='__main__':
    config = sweep_config()
    sweep_id = wandb.sweep(config, project='baseclassifier_paramtune_celebahq')
    wandb.agent(sweep_id=sweep_id, function=paramtune, count=100)

