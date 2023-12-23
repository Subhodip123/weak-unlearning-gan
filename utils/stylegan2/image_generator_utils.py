import sys
sys.path.append("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE")

# from torchvision.models import ResNet50_Weights
# from torchvision.models import ResNet50_Weights
import torch
import sys
import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn

from torchvision import transforms, utils
# from ebm import EBM

from tqdm import tqdm
@torch.no_grad()

def make_noise(batch, latent_dim, n_noise, device):
        if n_noise == 1:
                return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

        return noises


def mixing_noise(batch, latent_dim, prob, device):
        if prob > 0 and random.random() < prob:
                return make_noise(batch, latent_dim, 2, device)

        else:
                return [make_noise(batch, latent_dim, 1, device)]

device='cuda'


def img_classifier(images,noise,feature_type):
    device='cuda'
    resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
    resnext50_32x4d.fc = nn.Linear(2048, 40)
    resnext50_32x4d.to(device)
    path_toLoad="/home/ece/Subhodip/UnlearnGAN/weights_classifier/celebahq/model_3_epoch.pt"
    checkpoint = torch.load(path_toLoad)
    resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
    # global classifier_checkpoint
    
    # temp = []
    
    
    maxk = 1
    neg_image = []
    pos_image = []
    neg_noise = []
    noise=noise[0]
    # print(type(noise))
    # data = TensorDataset(images,noise)
    
    # print(data_loader.__dict__)
    # for data in data_loader:
    # 	data = data[0]
    images=images[0]
    with torch.no_grad():
          
        
        #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
        resnext50_32x4d.eval() 
        res=transforms.Resize((218,178))
        # ip=torch.randn(8,3,218,178).to(device)
        images=images.unsqueeze(0)
        images=res(images)
        # print(images.shape)
        scores=resnext50_32x4d(images)
        # labels=torch.zeros(8,40).to(device)
        converted_Score=scores.clone()
        converted_Score[converted_Score>=0]=1
        converted_Score[converted_Score<0]=0
        # print(converted_Score)
        converted_Score=converted_Score.t()
    all_preds = converted_Score
    feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    # print(type(feature_type))
    ind=feature_dict[feature_type]
    # print("feature index is:",ind)

    neg = all_preds[ind] #Index: 20; because we're suppresing Male
    if(feature_type=="No_Beard"):
        # print("yoooooooooooooooooooo")
        pos_index = torch.where(neg ==1)[0]
        neg_index = torch.where(neg == 0)[0]

    else:
         neg_index = torch.where(neg ==1)[0]
         pos_index = torch.where(neg == 0)[0]
    neg_image.append(images[neg_index])
    # neg_noise.append(noise[neg_index])
    # pos_noise.append(noise[pos])
    pos_image.append(images[pos_index])
    # neg_noise = torch.cat(neg_noise, dim=0)
    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    return neg_images, pos_images,neg_noise
    


def get_mean_style(generator, device):
    mean_style = None
    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

def gen_images_tot(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    loop_len=int(tot_samples/10)
    cnt=0

    # while(loop_len>cnt):
    #     cnt+=1

         
    for i in tqdm(range(loop_len)):
        z_latent=mixing_noise(10, 512, 0.9, 'cuda')
        # z_latent = torch.randn(100, 512).to(device)
        gen_img = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
        # print("Neg size", neg.size())
        # print("Pos size",neg.size()[0])

    if tot_samples%10!=0:


        z_latent = mixing_noise(tot_samples%10, 512, 0.9, 'cuda')
        gen_img = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
    
    return neg_images, pos_images

def gen_images_tot_pos(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []

    while(len(pos_images)<tot_samples):
        print(len(pos_images))
        z_latent=mixing_noise(10, 512, 0.9, 'cuda')
        # z_latent = torch.randn(100, 512).to(device)
        gen_img,_ = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        if(len(neg)!=0):
                
            neg_images.extend(neg)

        else:
            pos_images.extend(pos)
            # print("Neg size", neg.size())
            # print("Pos size",neg.size()[0])

    
    
    pos_images=pos_images[0:tot_samples]
    return None, pos_images

def gen_images_tot_neg(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    
    
         
    while(len(neg_images)<tot_samples):
        # print(len(neg_images))

        z_latent=mixing_noise(64, 512, 0.9, 'cuda')
        # z_latent = torch.randn(100, 512).to(device)
        gen_img,_ = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        # print("images not added",len(pos))

        if(len(neg)!=0):
            # print("images added",len(neg))  
            
            neg_images.extend(neg)
            print(len(neg_images))

        else:
            pos_images.extend(pos)
            # print("Neg size", neg.size())
            # print("Pos size",neg.size()[0])

    # if tot_samples%10!=0:


    #     z_latent = torch.randn(tot_samples%10, 512).to(device)
    #     gen_img = G(z_latent)

    #     neg,pos,neg_noise = img_classifier(gen_img,z_latent)
    #     neg, pos = neg.detach().cpu(), pos.detach().cpu()
    #     neg_images.extend(neg)
    #     pos_images.extend(pos)
    neg_images=neg_images[0:tot_samples]
    return neg_images, None


# def gen_images_custom(G,pos_samples=5000, neg_samples=5000):




def gen_images_custom(G,pos_samples=5000, neg_samples=5000, interpolate=True):
    device= "cuda"
    mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    
    while(len(pos_images)<pos_samples or len(neg_images)<neg_samples):
        z_latent = torch.randn(100, 512).to(device)
        gen_img = G(z_latent, step=6, alpha=1,mean_style=mean_style,style_weight=0.7)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
        # print("Neg size", neg.size())
        # print("Pos size",neg.size()[0])
        if(interpolate):

            req_no = 100 - neg.size()[0]
            # print(f"req_no {req_no}")
            neg_latents = []
            for i in range(req_no):
                weight = i / (req_no - 1)
                ind1 = random.randint(0,len(neg_noise)-1)
                ind2 = random.randint(0, len(neg_noise)-1)				
                interpolated_vector = torch.lerp(neg_noise[ind1], neg_noise[ind2], weight)
                neg_latents.append(interpolated_vector)
            neg_latent = torch.stack(neg_latents,dim=0)
            # print(neg_latent.shape, len(neg_latents))

            neg_gen_imgs = G(neg_latent, step=6, alpha=1,mean_style=mean_style,style_weight=0.7)
            i_neg,_,_ = img_classifier(neg_gen_imgs,neg_latent)
            i_neg = i_neg.detach().cpu()
            # print("i_neg size=", i_neg.size())
            neg_images.extend(i_neg)

        if(len(pos_images)>=pos_samples):
            pos_images = pos_images[:pos_samples]


        # print(len(pos_images),len(neg_images))
        if len(neg_images) >= neg_samples and len(pos_images) >= pos_samples:
            neg_images = neg_images[:neg_samples]
            pos_images = pos_images[:pos_samples]
            break
    print(len(neg_images),len(neg_images))
    return neg_images, pos_images




class FeedbackData_neg(data.Dataset):

    def __init__(self, G,feature_type,sampling_type=2,pos_samples=5000,neg_samples=5000,tot_samples=5000, ind=0):
        

        
        neg_images, pos_images = gen_images_tot(G,feature_type,tot_samples)

       

        # print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape, pos_images.shape)
        
        utils.save_image(
                        neg_images[0:64],
                        f"%s/{feature_type}/{str(len(neg_images)).zfill(6)}_{str(ind)}.png" % ("train_samples"),
                        nrow=int(len(neg_images[0:64]) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        self.neg_img = neg_images
        

    def __getitem__(self, index):
        neg = self.neg_img[index]
        
        


        return neg

        

    def __len__(self):
        return len(self.neg_img)
    


class FeedbackData_pos(data.Dataset):

    def __init__(self, G,feature_type,sampling_type=2,pos_samples=5000,neg_samples=5000,tot_samples=5000, ind=0):
        

        
        neg_images, pos_images = gen_images_tot(G,feature_type,tot_samples)

       

        # print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape, pos_images.shape)
        
        utils.save_image(
                        pos_images[0:64],
                        f"%s/{feature_type}/{str(len(pos_images)).zfill(6)}_{str(ind)}.png" % ("train_samples"),
                        nrow=int(len(neg_images[0:64]) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        self.pos_img = pos_images
        

    def __getitem__(self, index):
        pos = self.pos_img[index]
        
        


        return pos

        

    def __len__(self):
        return len(self.pos_img)