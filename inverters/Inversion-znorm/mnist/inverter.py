import torch
from torch import nn

class GANInverter(nn.Module):
    """Encoder class that is being learned"""

    def __init__(
        self, in_channels: int, latent_dim: int, gen_net: torch.nn.Module, last_layer_size:int, 
        hidden_dims: list = None, 
    ) -> None:
        """Encoder Class initalizer"""
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.gen_net = gen_net
        self.last_layer_size = last_layer_size
        # build the encoder
        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            self.in_channels = h_dim
        # final linear layers
        self.inverter = nn.Sequential(*modules)
        self.linear_layer = nn.Linear(self.hidden_dims[-1], self.last_layer_size)
        self.activation = nn.LeakyReLU()
        self.last_layer = nn.Linear(self.last_layer_size, latent_dim)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) gives the latent representation
        """
        result = self.inverter(input)
        result = torch.flatten(result, start_dim=1)
        result = self.linear_layer(result)
        result = self.activation(result)
        latent_representations = self.last_layer(result)
        return latent_representations

    def loss(self, input: torch.tensor, actual_noise: torch.tensor) -> float:
        """Define the loss function for the encoder"""
        latent_noise = self.forward(input)
        loss_on_batch = torch.sum(torch.norm(actual_noise-latent_noise, dim=1))
        return loss_on_batch