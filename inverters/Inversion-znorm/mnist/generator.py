import torch
from torch.utils.data import DataLoader

class DataGenerator(object):
    """Class for the data generation"""

    def __init__(
        self,
        pretrained_gan: torch.nn.Module,
        no_samples: int,
        batch_size: int,
        img_dims: tuple,
        latent_dim=100,
    ) -> None:
        """Class initiallizer
        img_dims: tuple of format(CxHxW)"""
        self.gen_net = pretrained_gan
        self.N = no_samples
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.latent_dim = latent_dim

    def generate(self) -> torch.tensor:
        """generates N no of samples from normal prior"""
        iter_no = int(self.N / self.batch_size)
        fake_images = []
        fixed_noises = []
        for iter in range(iter_no):
            fixed_noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
            if torch.cuda.is_available():
                fixed_noise = fixed_noise.cuda()
            fake_image = self.gen_net(fixed_noise)
            fake_image = fake_image.cpu().detach()
            fixed_noise = fixed_noise.cpu().detach()
            fake_images.append(fake_image)
            fixed_noises.append(fixed_noise)
        fake_images = torch.cat(fake_images, dim=0)
        fixed_noises = torch.cat(fixed_noises, dim=0)
        return fake_images, fixed_noises

    def train_val_test_split(self) -> tuple:
        """Make 3 partitions on generated images"""
        generated_tensor, noise_tensor = self.generate()
        train_loader = DataLoader(dataset=generated_tensor[0 : int(self.N * 0.8)], batch_size=self.batch_size,
                                  shuffle=False)
        val_loader = DataLoader(dataset=generated_tensor[int(self.N * 0.8) : int(self.N * 0.9)], 
                                 batch_size=self.batch_size, shuffle=False)
        test_tensors = generated_tensor[int(self.N * 0.9) :]
        train_noise_loader = DataLoader(dataset=noise_tensor[0 : int(self.N * 0.8)], batch_size=self.batch_size,
                                        shuffle=False)
        val_noise_loader = DataLoader(noise_tensor[int(self.N * 0.8) : int(self.N * 0.9)], batch_size=self.batch_size,
                                        shuffle = False)
        test_noise = noise_tensor[int(self.N * 0.9) :]
        return train_loader, train_noise_loader, val_loader, val_noise_loader, test_tensors, test_noise