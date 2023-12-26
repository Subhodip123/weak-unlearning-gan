import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
np.random.seed(44)


class DataAugment(object):
    """Reference class for Multiple data augmentation"""

    def __init__(self, org_img_tensor) -> None:
        self.org_img_tensor = org_img_tensor

    def horizontalflip_aug(self) -> list:
        """flips the original image horizontally"""
        
        hflipper = transforms.RandomHorizontalFlip(p=1)
        transformed_img = hflipper(self.org_img_tensor)
        return [transformed_img]

    def verticalflip_aug(self) -> list:
        """flips the original image vertically"""
        vflipper = transforms.RandomVerticalFlip(p=1)
        transformed_img = vflipper(self.org_img_tensor)
        return [transformed_img]


    def gaussian_blur(self) -> list:
        """Apply gaussian blurring on the image"""
        blurry_imgs = []
        for _ in range(6):
            blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2))
            blurry_imgs.append(blur(self.org_img_tensor))
        return blurry_imgs

    def perspective_aug(self) -> list:
        """Give the image different looking point"""
        perspective_imgs = []
        for _ in range(6):
            random_no = np.random.uniform()
            perspective_transformer = transforms.RandomPerspective(
                distortion_scale=random_no, p=1.0
            )
            perspective_imgs.append(perspective_transformer(self.org_img_tensor))
        return perspective_imgs

    def rotation_aug(self) -> list:
        """Gives random rotation to the image"""
        rotated_imgs =[]
        for _ in range(6):
            rotater = transforms.RandomRotation(degrees=(0, 180))
            rotated_imgs.append(rotater(self.org_img_tensor))
        return rotated_imgs

    def affine_aug(self) -> list:
        """gives random affine transformation to the images"""
        affine_imgs = []
        for _ in range(6):
            affine_transfomer = transforms.RandomAffine(
                degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
            )
            affine_imgs.append(affine_transfomer(self.org_img_tensor))
        return affine_imgs


    def sharpness_aug(self) -> list:
        """performs sharpness adjustment to the original image"""
        sharpened_imgs =[]
        for _ in range(6):
            sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=2)
            sharpened_imgs.append(sharpness_adjuster(self.org_img_tensor))
        return sharpened_imgs

    def auto_aug(self) -> list:
        """This gives the auto augmented images"""
        autoaug_imgs =[]
        policy = transforms.autoaugment.AutoAugmentPolicy.SVHN
        for _ in range(31):
            augmentator = transforms.AutoAugment(policy = policy)
            autoaug_imgs.append((augmentator(self.org_img_tensor.type(torch.uint8))).type(torch.float))
        return autoaug_imgs

    def total_augment(self) -> torch.tensor:
        """Given a list of tensors it will stack them in a new dimension"""
        # get different augmentation of data
        augmented_references = [self.org_img_tensor]
        horizontal_flip_imgs = self.horizontalflip_aug()
        vertical_flip_imgs = self.verticalflip_aug()
        blurry_imgs = self.gaussian_blur()
        perspective_imgs = self.perspective_aug()
        rotated_imgs = self.rotation_aug()
        affine_imgs = self.affine_aug()
        sharpened_imgs = self.sharpness_aug()
        autoaug_imgs = self.auto_aug()
        augmented_references.extend(horizontal_flip_imgs)
        augmented_references.extend(vertical_flip_imgs)
        augmented_references.extend(blurry_imgs)
        augmented_references.extend(perspective_imgs)
        augmented_references.extend(rotated_imgs)
        augmented_references.extend(affine_imgs)
        augmented_references.extend(sharpened_imgs)
        augmented_references.extend(autoaug_imgs)
        augmented_references_tensor = torch.concat(augmented_references, dim=0).type(torch.float)
        return augmented_references_tensor


def augment_tensorization(self, data_tensors:torch.Tensor):
        """given the directory of user ref images it reads all the images into a tensor"""
        aug_img_tensors = []
        for i in range(len(data_tensors)):
            org_img_tensor = data_tensors[i]
            data_aug = DataAugment(org_img_tensor=org_img_tensor)
            data_aug_tensor = data_aug.total_augment()
            aug_img_tensors.append(data_aug_tensor)
        aug_img_tensors = torch.concat(aug_img_tensors, dim=0)
        return aug_img_tensors
