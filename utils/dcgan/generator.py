import torch
from torch.utils.data import DataLoader


class DataGenerator(object):
    """Class for the data generation"""

    def __init__(
        self,
        pretrained_gan: torch.nn.Module,
        no_samples: int,
        img_dims: tuple,
        batch_size: int,
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
        for iter in range(iter_no):
            fixed_noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
            if torch.cuda.is_available():
                fixed_noise = fixed_noise.cuda()
            fake_image = self.gen_net(fixed_noise)
            fake_image = fake_image.cpu().detach()
            fake_images.append(fake_image)
        fake_images = torch.cat(fake_images, dim=0)
        return DataLoader(fake_images, batch_size=self.batch_size), fake_images

