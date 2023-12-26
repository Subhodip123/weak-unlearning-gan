import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from augment import DataAugment
from PIL import Image

class Data(object):
    """Reference class for User inputs"""

    def __init__(self, pos_path, neg_path, length) -> None:
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.length = length

    def reference_tensorization(self, path, augment=False):
        """given the directory of user ref images it reads all the images into a tensor"""
        ref_img_tensors = []
        transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
        for filename in os.listdir(path)[:self.length]:
            org_img = Image.open(os.path.join(path, filename))
            img_tensor = (transformations(org_img.convert('RGB')) *255).type(torch.float)[None,:,:,:]
            if augment == True:
                img_augment = DataAugment(img_tensor)
                img_tensor = img_augment.total_augment()
            ref_img_tensors.append(img_tensor)
        ref_img_tensors = torch.concat(ref_img_tensors, dim=0)
        ref_img_tensors = (ref_img_tensors - 127.5)/127.5
        print('hi',ref_img_tensors.shape, torch.max(ref_img_tensors), torch.min(ref_img_tensors))
        return ref_img_tensors
    
    def dataloader(self, augment):
        "it gives the data loader"
        pos_data = self.reference_tensorization(self.pos_path, augment=augment)
        neg_data = self.reference_tensorization(self.neg_path, augment=augment)
        pos_labels = torch.ones((len(pos_data),), dtype=int)
        neg_labels = torch.zeros((len(neg_data),), dtype=int)
        #train and val data
        pos_train_data = pos_data[:int(len(pos_data)*.8)]
        neg_train_data = neg_data[:int(len(neg_data)*.8)]
        pos_val_data = pos_data[int(len(pos_data)*.8):]
        neg_val_data = neg_data[int(len(neg_data)*.8):]
        train_data = torch.cat([pos_train_data, neg_train_data], dim=0)
        val_data = torch.cat([pos_val_data, neg_val_data], dim=0)
        # train and val labels
        pos_train_labels = pos_labels[:int(len(pos_labels)*.8)]
        neg_train_labels = neg_labels[:int(len(neg_labels)*.8)]
        pos_val_labels = pos_labels[int(len(pos_data)*.8):]
        neg_val_labels = neg_labels[int(len(neg_data)*.8):]
        train_labels = torch.cat([pos_train_labels, neg_train_labels], dim=0)
        val_labels = torch.cat([pos_val_labels, neg_val_labels], dim=0)
        #dataloaders
        if augment == False:
            train_dataloader = DataLoader(train_data, batch_size=len(train_data))
            val_dataloader = DataLoader(val_data, batch_size=len(val_data))
            train_labelloader = DataLoader(train_labels,batch_size=len(train_labels))
            val_labelloader = DataLoader(val_labels, batch_size=len(val_labels))
        else:
            train_dataloader = DataLoader(train_data, batch_size=64)
            val_dataloader = DataLoader(val_data, batch_size=64)
            train_labelloader = DataLoader(train_labels,batch_size=64)
            val_labelloader = DataLoader(val_labels, batch_size=64)
        
        return train_dataloader, val_dataloader, train_labelloader, val_labelloader
