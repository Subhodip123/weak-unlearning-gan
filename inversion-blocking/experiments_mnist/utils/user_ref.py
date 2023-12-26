import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class DataAugment(object):
    """Reference class for data augmentation"""

    def __init__(self, org_img) -> None:
        self.org_img = org_img

    def gaussian_blur(self) -> list:
        """Apply gaussian blurring on the image"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 3))
        blurry_imgs = [blur(img_tensor) for _ in range(100)]
        return blurry_imgs

    def perspective_aug(self) -> list:
        """Give the image different looking point"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        perspective_imgs = []
        for _ in range(100):
            random_no = np.random.uniform()
            perspective_transformer = transforms.RandomPerspective(
                distortion_scale=random_no, p=1.0
            )
            perspective_imgs.append(perspective_transformer(img_tensor))
        return perspective_imgs

    def rotation_aug(self) -> list:
        """Gives random rotation to the image"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        rotater = transforms.RandomRotation(degrees=(0, 180))
        rotated_imgs = [rotater(img_tensor) for _ in range(100)]
        return rotated_imgs

    def affine_aug(self) -> list:
        """gives random affine transformation to the images"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        affine_transfomer = transforms.RandomAffine(
            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
        )
        affine_imgs = [affine_transfomer(img_tensor) for _ in range(100)]
        return affine_imgs

    def sharpness_aug(self) -> list:
        """performs sharpness adjustment to the original image"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=2)
        sharpened_imgs = [sharpness_adjuster(img_tensor) for _ in range(100)]
        return sharpened_imgs

    def horizontalflip_aug(self) -> list:
        """flips the original image horizontally"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        hflipper = transforms.RandomHorizontalFlip(p=1)
        transformed_img = hflipper(img_tensor)
        return [transformed_img]

    def verticalflip_aug(self) -> list:
        """flips the original image vertically"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        vflipper = transforms.RandomVerticalFlip(p=1)
        transformed_img = vflipper(img_tensor)
        return [transformed_img]

    def augmented_tensorization(self) -> torch.tensor:
        """Given a list of tensors it will stack them in a new dimension"""
        img_tensor = (transforms.ToTensor()(self.org_img.convert('L')) * 255).type(
            torch.int
        )
        # get different augmentation of data
        augmented_references = [img_tensor]
        blurry_imgs = self.gaussian_blur()
        perspective_imgs = self.perspective_aug()
        rotated_imgs = self.rotation_aug()
        affine_imgs = self.affine_aug()
        sharpened_imgs = self.sharpness_aug()
        horizontal_flip_imgs = self.horizontalflip_aug()
        vertical_flip_imgs = self.verticalflip_aug()
        augmented_references.extend(blurry_imgs)
        augmented_references.extend(perspective_imgs)
        augmented_references.extend(rotated_imgs)
        augmented_references.extend(affine_imgs)
        augmented_references.extend(sharpened_imgs)
        augmented_references.extend(horizontal_flip_imgs)
        augmented_references.extend(vertical_flip_imgs)
        augmented_references_tensor = torch.stack(augmented_references).type(torch.float32)
        return augmented_references_tensor


class UserMultiRef(object):
    """Reference class for User inputs"""

    def __init__(self, path, length) -> None:
        self.path = path
        self.length = length

    def reference_tensorization(self, augment=False):
        """given the directory of user ref images it reads all the images into a tensor"""
        ref_img_tensors = []
        for filename in os.listdir(self.path)[:self.length]:
            org_img = Image.open(os.path.join(self.path, filename))
            if augment:
                data_aug = DataAugment(org_img=org_img)
                img_tensor = data_aug.augmented_tensorization()
                ref_img_tensors.append(img_tensor)
            else:
                img_tensor = (transforms.ToTensor()(org_img.convert('L')) *255).type(torch.float)
                ref_img_tensors.append(img_tensor)
        ref_img_tensors = torch.stack(ref_img_tensors)
        return ref_img_tensors
