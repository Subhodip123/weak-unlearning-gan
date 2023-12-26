import os
import warnings
import torch
import wandb
from config import sweep_config
from utils.dcgan import Generator
from generator import DataGenerator
from inverter import GANInverter
from traininverter import TrainInverter

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
torch.manual_seed(44)



def paramtune_inverter(config=None):
    """Hyper parameter tuning for inverter"""

    #--------------Make results directory-----------#
    if os.path.exists('UnlearnGAN/Unlearn-Blocking/inverters/Inversion-znorm/mnist/results')==False:
        os.makedirs('UnlearnGAN/Unlearn-Blocking/inverters/Inversion-znorm/mnist/results')
    #-----------------------------------------------#

    #-------------Generator-----------------#
    G = Generator(ngpu=1).eval()
    G.requires_grad_(False)
    # load weights
    G.load_state_dict(torch.load("/home/ece/Subhodip/UnlearnGAN/weights_gan/dcgan_checkpoints/mnist/netG_epoch_99.pth"))
    if torch.cuda.is_available():
        G = G.cuda()
    #-------------------------------------------#

    #---------------Inverter Training---------------------#
    latent_dim = 100
    with wandb.init(config=config):
        config = wandb.config
        inverter_model = GANInverter(in_channels=1, latent_dim=latent_dim, gen_net=G, 
                                     last_layer_size=config.last_layer)
        inverter_model = inverter_model.to(device=device)
        optimizer = torch.optim.Adamax(
            inverter_model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        datagen = DataGenerator(pretrained_gan=G, no_samples=51200, batch_size = config.batch_size, 
                                img_dims=(1, 28, 28))
        trainer = TrainInverter(
            data_generator=datagen,
            inverter=inverter_model,
            epochs=config.epochs,
            optimizer=optimizer,
            early_stop=config.early_stop
        )
        trainer.training()
    

if __name__ == '__main__':
    
    wandb.login()
    config = sweep_config()
    sweep_id = wandb.sweep(config, project='inverterznorm_paramtune_mnist')
    wandb.agent(sweep_id=sweep_id, function=paramtune_inverter, count=50)
