import torch
from torch.utils.data import DataLoader
from torchvision import utils
from utils.model import Generator 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class DataGenerator(object):
    """Class for the data generation"""

    def __init__(
        self,
        checkpoiint: str,
        no_samples: int,
        output_size: int,
        latent_dim: int,
        
    ) -> None:
        """Class initiallizer
        img_dims: tuple of format(CxHxW)"""
        self.N = no_samples
        self.output_size = output_size
        self.latent_dim = latent_dim
        n_mlp = 8
        g_ema = Generator( self.output_size, self.latent_dim, n_mlp).to(device)
        checkpoint = torch.load(checkpoiint)
        g_ema.load_state_dict(checkpoint["g_ema"],strict=True)
        self.gen_net = g_ema
        
    # def generate(self, g_ema, device):
    #     """generates the fake images"""
    #     with torch.no_grad():
    #         g_ema.eval()
    #         images = []
    #         latents = []
    #         for i in range(self.N):
    #             sample_z = torch.randn(1, self.latent_dim, device=device)
    #             image, latent = g_ema.forward(
    #                 [sample_z])
    #             image = image.cpu().detach()
    #             latent = latent.cpu().detach()
    #             images.append(image)
    #             latents.append(latent)
    #         images = torch.concat(images, dim=0)
    #         latents = torch.concat(latents, dim=0)
    #     return images, latents
    
    def generate(self, g_ema, device):
        """generates the fake images"""
        with torch.no_grad():
            g_ema.eval()
            images = []
            latents = []
            for i in range(self.N):
                sample_z = torch.randn(1, self.latent_dim, device=device)
                mean_latent=g_ema.style(sample_z).mean(0,keepdim=True)
                image, latent = g_ema.forward(
                    [sample_z])
                sample_z=sample_z.cpu().detach()
                mean_latent=mean_latent.cpu().detach()
                image = image.cpu().detach()
                latent = latent.cpu().detach()
                images.append(image)
                latents.append(mean_latent)
            
            

            images = torch.concat(images, dim=0)
            latents = torch.concat(latents, dim=0)
        return images, latents


    def train_test_split(self) -> tuple:
        """Make 3 partitions on generated images"""
        generated_tensor, noise_tensor = self.generate(g_ema=self.gen_net, device=device)
        train_data_tensors = generated_tensor[0 : int(self.N * 0.7)]
        test_data_tensors = generated_tensor[int(self.N * 0.7) :]
        train_noise_tensors = noise_tensor[0 : int(self.N * 0.7)]
        test_noise_tensors = noise_tensor[int(self.N * 0.7) :]
        return train_data_tensors, train_noise_tensors, test_data_tensors, test_noise_tensors

