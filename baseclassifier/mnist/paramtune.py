import os
import torch
import wandb
from torch.optim import Adam, Adamax
import numpy as np
from datagen import Data
from model import BaseClassifier
from config import sweep_config
from train import training
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if os.path.exists('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results/class-5') == False:
    os.makedirs('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results/class-5')

pos_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/mnist/class-5"
neg_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/mnist/class-5"

wandb.login()


def paramtune(config=None):
    """Trains the classifier"""
    length = 50
    with wandb.init(config=config):
       config = wandb.config
       data = Data(pos_path=pos_path, neg_path=neg_path, length=length)
       train_dataloader, val_dataloader, train_labelloader, val_labelloader = data.dataloader(augment=True)
       model = BaseClassifier(inchannels=1)
       model = model.to(device=device)
       optimizer = Adamax([param for param in model.parameters() if param.requires_grad == True ], lr = config.lr, betas=(config.beta1, config.beta2))
       no_epochs = int(config.epochs)
       training(model=model, model_optim=optimizer, no_epochs=no_epochs, train_dataloader=train_dataloader,
                train_labelloader=train_labelloader, val_dataloader=val_dataloader,
                val_labelloader=val_labelloader, length=length, early_stop=config.early_stop)


if __name__=='__main__':
    config = sweep_config()
    sweep_id = wandb.sweep(config, project='augbaseclassifier_paramtune_mnist')
    wandb.agent(sweep_id=sweep_id, function=paramtune, count=100)