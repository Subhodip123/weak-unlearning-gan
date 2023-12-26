import os
import torch
from torch.optim import Adam,Adamax
import numpy as np
import wandb
from datagen import Data
from config import config, augconfig
from model import BaseClassifier
from train import training
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if os.path.exists('/home/ece/Subhodip/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results') == False:
    os.makedirs('/home/ece/Subhodip/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results')

pos_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/celebahq/class-bangs"
neg_path = "/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_neg_imgs/celebahq/class-bangs"



def run_log(config=None, length=None):
    """Trains classifier with best parameter for one run"""
    wandb.init(project='optimized_baseclassifier_celebahq_mullengths', name=f'run_{length}', config=config)
    config = wandb.config
    data = Data(pos_path=pos_path, neg_path=neg_path, length=length)
    train_dataloader, val_dataloader, train_labelloader, val_labelloader = data.dataloader(augment=False)
    model = BaseClassifier(inchannels=3)
    model = model.to(device=device)
    optimizer = Adamax([param for param in model.parameters() if param.requires_grad == True ], lr = config.lr, betas=(config.beta1, config.beta2))
    training(model=model, model_optim=optimizer, no_epochs=config.epochs, train_dataloader=train_dataloader,
            train_labelloader=train_labelloader, val_dataloader=val_dataloader,
            val_labelloader=val_labelloader, length=length, early_stop=config.early_stop)
    wandb.finish()

def main():
    """Trains the classifier with best parameter"""
    lengths = np.arange(10,101,10).tolist() #+ np.arange(200,1001,200).tolist() + [2000,3000,4000,5000]  
    for length in lengths:
        print('Starting',length)
        config_dic = config()
        run_log(config=config_dic, length=length)    
        

if __name__=='__main__':
    wandb.login()
    main()


