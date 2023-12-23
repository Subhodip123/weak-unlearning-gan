import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from dcgan_mnist import Generator
from dcgan_cifar import Generator as Generator1
from lenet import LeNet5
from classifier import cifar10
from generator import DataGenerator
np.random.seed(44)
torch.manual_seed(44)


def user_ref_pos_neg(data, class_no:int, pos_path:str, neg_path:str, no_pos:int, no_neg:int):
    """Given the path of the data with labels it will save positive and negetive images"""
    if (os.path.exists(path=pos_path), os.path.exists(path=neg_path)) == (False, False):
        os.makedirs(pos_path)
        os.makedirs(neg_path)
    if data == 'mnist':
        #-------------Generate Data-----------------#
        G = Generator(ngpu=1).requires_grad_(False)
        G.load_state_dict(torch.load("/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_gan/dcgan_checkpoints/mnist/netG_epoch_99.pth"))
        if torch.cuda.is_available():
            G = G.cuda()
        datagen = DataGenerator(pretrained_gan=G, no_samples=12800, batch_size = 128, img_dims=(1, 28, 28))
        gen_data_loader, fake_imgs  = datagen.generate()
        # print(fake_imgs.shape)
        # save_image(fake_imgs[:100], 'generated.png', nrow=20, normalize=True, value_range=(-1,1))
        #-------------------------------------------#
        # load classifier
        org_classifiernet = LeNet5().eval()
        org_classifiernet.load_state_dict(torch.load("/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_classifier/mnist/lenet_epoch=12_test_acc=0.991.pth"))
        org_classifiernet = org_classifiernet.to(device='cuda:0')
        # prediction of negative and posotive data
        predicted_labels = []
        for gen_batch_data in gen_data_loader:
            gen_batch_data = gen_batch_data.to(device='cuda:0')
            gen_batch_data = transforms.Resize((32, 32))(gen_batch_data)
            pred_labels = org_classifiernet(gen_batch_data).cpu().detach().numpy()
            pred_class = np.argmax(pred_labels, axis=1).tolist()
            predicted_labels.extend(pred_class)
            gen_batch_data.to('cpu')
        pred_class_tensor = torch.tensor(data=predicted_labels)
        pred_refclass_loc = torch.where(pred_class_tensor==class_no)[0][:no_pos]
        pred_nonrefclass_loc = torch.where(pred_class_tensor!=class_no)[0][:no_neg]
        pos_imgs = fake_imgs[pred_refclass_loc]
        neg_imgs = fake_imgs[pred_nonrefclass_loc]
        for i, img_tensor in enumerate(pos_imgs):
            save_image(img_tensor, pos_path+'/'+str(i)+'.png',normalize=True, value_range=(-1,1))
        for j, img_tensor in enumerate(neg_imgs):
            save_image(img_tensor, neg_path+'/'+str(j)+'.png',normalize=True, value_range=(-1,1))
            
    # elif data=='cifar10':
    #     #-------------Generate Data-----------------#
    #     G = Generator1(ngpu=1).requires_grad_(False)
    #     # load weights
    #     G.load_state_dict(torch.load("/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_gan/dcgan_checkpoints/cifar10/netG_epoch_199.pth"))
    #     if torch.cuda.is_available():
    #         G = G.cuda()
    #     datagen = DataGenerator(pretrained_gan=G, no_samples=12800, batch_size = 128, 
    #                             img_dims=(3, 32, 32))
    #     gen_data_loader, fake_imgs = datagen.generate()
    #     #-------------------------------------------#
    #     # load classifier
    #     classifier = cifar10(128, pretrained='log/cifar10/best-135.pth')
    #     net = classifier.eval()
    #     net = net.to(device='cuda:0')
    #     # prediction of negative and posotive data
    #     predicted_labels = []
    #     for gen_batch_data in gen_data_loader:
    #         gen_batch_data = gen_batch_data.to(device='cuda:0')
    #         pred_labels = org_classifiernet(gen_batch_data).cpu().detach().numpy()
    #         pred_class = np.argmax(pred_labels, axis=1).tolist()
    #         predicted_labels.append(pred_class)
    #         gen_batch_data.to('cpu')
    #     pred_class_tensor = torch.tensor(data=predicted_labels)
    #     pred_refclass_loc = torch.where(pred_class_tensor==class_no)[0][:no_pos]
    #     pred_nonrefclass_loc = torch.where(pred_class_tensor!=class_no)[0][:no_neg]
    #     pos_imgs = fake_imgs[pred_refclass_loc]
    #     neg_imgs = fake_imgs[pred_nonrefclass_loc]
    #     for i, img_tensor in enumerate(pos_imgs):
    #         save_image(img_tensor, pos_path+'/'+str(i)+'.png',normalize=True, value_range=(-1,1))
    #     for j, img_tensor in enumerate(neg_imgs):
    #         save_image(img_tensor, neg_path+'/'+str(j)+'.png',normalize=True, value_range=(-1,1))

    return None

if __name__ == '__main__':
    data = 'mnist'
    class_no = 8
    pos_path = '/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_pos_imgs/mnist/class-8'
    neg_path = '/home/ece/Subhodip/Unlearning/UnlearnGAN/user_ref_neg_imgs/mnist/class-8'
    no_pos = 100
    no_neg = 100
    user_ref_pos_neg(data = data, class_no=class_no, pos_path=pos_path, neg_path=neg_path, 
                     no_pos=no_pos, no_neg=no_neg)
