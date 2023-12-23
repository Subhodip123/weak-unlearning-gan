import os
import numpy as np
import torch
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms, datasets,utils
# import torchvision
from model import Generator
from image_generator_utils import gen_images_tot_neg, gen_images_tot_pos
torch.manual_seed(44)




def user_ref_pos_neg(data, feature: str, pos_path:str, neg_path:str):
    """Given the path of the data with labels it will save positive and negetive images"""
    if (os.path.exists(path=pos_path), os.path.exists(path=neg_path)) == (False, False):
        os.makedirs(pos_path)
        os.makedirs(neg_path)
    if data=='celebahq':
        generator = Generator(
        256,512,8, channel_multiplier=2
    ).to('cuda')
        ckpt=torch.load("/home/ece/Subhodip/UnlearnGAN/weights_gan/stylegan2_checkpoints/CelebAHQ_checkpoint/370000.pt")
        generator.load_state_dict(ckpt["g_ema"],strict=True)
        
        _,neg_imgs=gen_images_tot_pos(G=generator,feature_type=feature,tot_samples=100)
        # print(len(pos_imgs))
        for i, img_tensor in enumerate(neg_imgs):
            print(img_tensor.shape)
            # img_tensor=img_tensor.unsqueeze(0)
            utils.save_image(img_tensor, neg_path+'/'+str(i)+'.png',normalize=True)
            # img = Image.fromarray(img_tensor.numpy(), mode='RGB')
            # img.save(os.path.join(pos_path,str(i)+'.png'))
        pos_imgs,_= gen_images_tot_neg(generator,feature,100)
        # print(len(neg_imgs))
        
        for j, img_tensor in enumerate(pos_imgs):
            # img_tensor=img_tensor.unsqueeze(0)
            utils.save_image(img_tensor,pos_path+'/'+str(j)+'.png',normalize=True)
            # img = Image.fromarray(img_tensor.numpy(), mode='RGB')
            # img.save(os.path.join(neg_path, str(j)+ '.png'))
        


    return None

if __name__ == '__main__':
    data = 'celebahq'
    feature = "Wearing_Hat"
    pos_path = '/home/ece/Subhodip/UnlearnGAN/user_ref_pos_imgs/celebahq/class-hats'
    neg_path = '/home/ece/Subhodip/UnlearnGAN/user_ref_neg_imgs/celebahq/class-hats'
    user_ref_pos_neg(data = data, feature=feature , pos_path=pos_path, neg_path=neg_path)
