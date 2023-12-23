from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
# import dlib
from torchvision.utils import save_image
from model import Generator
import torchvision.transforms as transforms
from models.psp import pSp
trans = transforms.Compose([
    transforms.Resize((256, 256)),
    
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
resize_dims = (256, 256)

model_path="/home/ece/Subhodip/UnlearnGAN/Unlearn-Blocking/inverters/Inversion-xnorm/celebahq/encoder4editing/e4e_ffhq_encode.pt"
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
# pprint.pprint(opts)  # Display full options used
# update the training options
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
print(opts.stylegan_size)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')
def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    
    return images, latents
# def run_alignment(image_path):
#   from encoder4editing.utils.alignment import align_face
#   predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#   aligned_image = align_face(filepath=image_path, predictor=predictor) 
#   print("Aligned image has shape: {}".format(aligned_image.size))
#   return aligned_image 

generator = Generator(
        256, 512, 8, channel_multiplier=2
    ).to("cuda")

ckpt=torch.load("/home/ece/Subhodip/UnlearnGAN/weights_gan/stylegan2_checkpoints/CelebAHQ_checkpoint/370000.pt")
generator.load_state_dict(ckpt["g_ema"])

z_latent=torch.randn(1,512).to('cuda')
image,_=generator([z_latent])
save_image(image[0],"actual.png")

image=trans(image)
with torch.no_grad():
   images,latents=run_on_batch(image,net)



print(images[0].shape)
save_image(images[0],"fake.png")
print(latents.shape)


