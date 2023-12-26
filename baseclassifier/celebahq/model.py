import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet50, ResNet50_Weights

class BaseClassifier(nn.Module):
    """Model for the classifier"""
    def __init__(self, inchannels: int) -> None:
        super().__init__()
        self.inchannels = inchannels
        weights = ResNet50_Weights.DEFAULT
        self.encoder = resnet50(weights=weights, progress=False)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features,1024)
        # self.batchnorm = nn.BatchNorm1d(1024)
        self.encoder_activation = nn.LeakyReLU()
        self.last_layer1 = nn.Linear(1024, 256)
        # self.batchnorm1 = nn.BatchNorm1d(256)
        self.last_layer1_act = nn.LeakyReLU()
        self.last_layer2 = nn.Linear(256,64)
        # self.batchnorm2 = nn.BatchNorm1d(64)
        self.last_layer2_act = nn.LeakyReLU()
        self.last_layer3 = nn.Linear(64,16)
        # self.batchnorm3 = nn.BatchNorm1d(16)
        self.last_layer3_act = nn.LeakyReLU()
        self.last_layer4 = nn.Linear(16,2)
        self.last_activation = nn.Softmax()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) gives the latent representation
        """
        if self.inchannels == 1:
            input = torch.cat([input, input, input], dim=1)
        result = self.encoder(input)
        # print("Result from Encoder: ", result.shape)
        #result = self.batchnorm(result)
        result = self.encoder_activation(result)
        result = self.last_layer1(result)
        #result = self.batchnorm1(result)
        result = self.last_layer1_act(result)
        result = self.last_layer2(result)
        #result = self.batchnorm2(result)
        result = self.last_layer2_act(result)
        result = self.last_layer3(result)
        #result = self.batchnorm3(result)
        result = self.last_layer3_act(result)
        result = self.last_layer4(result)
        return result
    
    def predict(self, input: torch.tensor):
        """Predict the labels """
        result = self.forward(input)
        result = self.last_activation(result)
        # print(result)
        labels = torch.argmax(result, dim=1)
        return labels
    
    def loss(self, input: torch.tensor ,labels: torch.tensor):
        """Loss funciton for the classifier"""
        pred_labels = self.forward(input=input)
        # pred_labels = pred_labels.reshape((pred_labels.size()[0],))
        loss = CrossEntropyLoss()(pred_labels, labels)
        return loss
