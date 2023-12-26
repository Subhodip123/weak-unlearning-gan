import numpy as np
import torch
from torchvision.utils import save_image
from torchvision import transforms

def recon_orig(device, inverter_path, generator, real_image_tensor, filename=None):
    """Checks the qualtity of the actual vs reconstructed image"""
    inverter = torch.load(inverter_path)
    real_image_tensor = real_image_tensor.to(device)
    latent_tensor = inverter.forward(real_image_tensor)
    latent_tensor = latent_tensor.reshape((latent_tensor.size()[0],latent_tensor.size()[1],1,1))
    recon_image = generator(latent_tensor)
    
    images = torch.cat([real_image_tensor[:100], recon_image[0:100]], dim=0)
    if filename:
        save_image(images, filename, nrow=20, normalize=True, value_range=(-1,1))
    else:
        save_image(images, "./results/real_vs_recon4.png", nrow=20, normalize=True, value_range=(-1,1))


