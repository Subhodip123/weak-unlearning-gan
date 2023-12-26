import torch
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def model_classifier(checkpoint_path):
    """Describing the model architecture to initialise from checkpoint for the pretrained classifier"""
    
    resnext50_32x4d = models.resnext50_32x4d(weights=True, progress=False)
    resnext50_32x4d.fc = nn.Linear(2048, 40)
    ct = 0
    for child in resnext50_32x4d.children():
        ct += 1
        if ct < 6:
            for param in child.parameters():
                param.requires_grad = False

    resnext50_32x4d.to(device)  
    checkpoint = torch.load(checkpoint_path)
    #Initializing the model with the model parameters of the checkpoint.
    resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
    #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
    resnext50_32x4d.eval()
    return resnext50_32x4d