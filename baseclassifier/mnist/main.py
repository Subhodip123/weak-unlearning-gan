import os
import torch
import wandb
from torch.optim import Adam, Adamax
import numpy as np
from datagen import Data
from model import BaseClassifier
from config import config, augconfig
from train import training
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if os.path.exists('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results') == False:
    os.makedirs('/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results')

pos_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/mnist/class-5"
neg_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/mnist/class-5"

def run_log(config=None, length=None):
    """Trains classifier with best parameter for one run"""
    wandb.init(project='optimized_baseclassifiers_mnist_mullengths', name=f'run_{length}', config=config)
    config = wandb.config
    data = Data(pos_path=pos_path, neg_path=neg_path, length=length)
    train_dataloader, val_dataloader, train_labelloader, val_labelloader = data.dataloader(augment=True) 
    model = BaseClassifier(inchannels=1)
    model = model.to(device=device)
    optimizer = Adamax([param for param in model.parameters() if param.requires_grad == True ], lr = config.lr, betas=(config.beta1, config.beta2))
    training(model=model, model_optim=optimizer, no_epochs=config.epochs, train_dataloader=train_dataloader,
            train_labelloader=train_labelloader, val_dataloader=val_dataloader,
            val_labelloader=val_labelloader, length=length, early_stop=config.early_stop)
    wandb.finish()

def main():
    """Trains the classifier with best parameter"""
    lengths = np.arange(10,101,10).tolist() 
    for length in lengths:
        print('Starting',length)
        config_dic = augconfig()
        run_log(config=config_dic, length=length)    
        

if __name__=='__main__':
    wandb.login()
    main()

