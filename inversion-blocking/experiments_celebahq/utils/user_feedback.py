import os
import numpy as np
import torch
import sys
sys.path.append("/home/ece/Subhodip/UnlearnGAN/Unlearn-Blocking/Inversion-Blocking/experiments_celebahq/utils")
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from argparse import Namespace
import time
import os
import sys
import numpy as np
from PIL import Image
import torch
# from models.psp import pSp
# import dlib
from torchvision.utils import save_image
from torch.utils.data import DataLoader
# from model import Generator
import torchvision.transforms as transforms
np.random.seed(44)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(44)


class UserFeedback(object):
    """Reference class for User feedback"""

    def __init__(self, data_tensor, inverter, classifier, classno, length, pos_neg=True) -> None:
        self.data_tensor = data_tensor
        self.inverter = inverter
        # ckpt = torch.load(inverter_path, map_location='cpu')
        # opts = ckpt['opts']
        # # pprint.pprint(opts)  # Display full options used
        # # update the training options
        # opts['checkpoint_path'] = inverter_path
        # opts= Namespace(**opts)
        # self.inverter = pSp(opts)
        # self.inverter.eval()
        # self.inverter.cuda()
        # self.inverter = torch.load(inverter_path).to(device=device)
        self.classifier = classifier
        self.length = length
        self.classno = classno
        self.pos_neg = pos_neg

    def projection_similarity(self, z_target: torch.tensor, z_input: torch.tensor):
            """calculates the projection similarity between the reference img and input"""
            similarity_arr = torch.zeros(size=(z_input.size()[0],))
            # print('a',z_input.size())
            # print('b',z_ref.size())
            batch_size = z_input.size()[0]
            for i in range(batch_size):
                similarity_score = torch.dot(z_target, z_input[i, :]) / torch.norm(z_target) 
                # print(similarity_score, i)
                similarity_arr[i] = similarity_score 
            return similarity_arr
        

    def mine_neg_examples(self, pos_noise_tensor:torch.tensor) -> torch.tensor:
        """Given the inverter, generated tensor, and positive user"""
        z_pos = pos_noise_tensor
        data_loader = DataLoader(self.data_tensor, batch_size=5)
        w_styles = []
        for batch_data in data_loader:
            batch_data=batch_data.to(device)
            _, w_batch_styles = self.inverter(batch_data.float(), randomize_noise=False, return_latents=True)
            w_styles.append(w_batch_styles.detach().cpu())
            batch_data.cpu().detach()
        w_styles = torch.concat(w_styles, dim=0)
        z_gen = torch.mean(w_styles, dim=1, keepdim=False)
        z_pos_mean = torch.mean(z_pos, dim=0)
        pos_sim_arr = self.projection_similarity(z_pos_mean, z_pos)
        generated_sim_arr = self.projection_similarity(z_pos_mean, z_gen)
        neg_loc = torch.where(generated_sim_arr<torch.min(pos_sim_arr))[0]
        random_selected_locations = np.random.choice(neg_loc.tolist(), size=len(z_pos))
        gen_neg_noise = z_gen[torch.tensor(random_selected_locations)]
        return gen_neg_noise
    
    def feedback_save(self, input, path):
        """saves the feedback image given by the user"""
        for i in range(self.length):
            img_tensor = input[i]
            save_image(img_tensor, f"{path}/{i}.png") 

    def feedback_tensorization(self):
        """given the directory of user ref images it reads all the images into a tensor"""
        gen_data = transforms.Resize((218, 178))(self.data_tensor)
        # print(gen_data.size())
        gen_dataloader=DataLoader(gen_data,batch_size=10)
        score=[]
        for data in gen_dataloader:
            data = data.to(device)
        # pred_labels = self.classifier(gen_data).cpu().detach().numpy()
            score.append(self.classifier(data).cpu().detach())
            data.cpu().detach()
        score=torch.cat(score,dim=0)
        # print(gen_data.shape,score.shape)
        converted_score=score.clone()
        converted_score[converted_score>=0]=1
        converted_score[converted_score<0]=0
        converted_score=converted_score.t()
        # print(gen_data.size())
        # pred_labels = self.classifier(gen_data).cpu().detach().numpy()
        # pred_class = np.argmax(pred_labels, axis=1)
        # pred_class_tensor = torch.tensor(data=pred_class)
        if self.pos_neg:
            pos=converted_score[self.classno]
            pred_refclass_loc=torch.where(pos==1)[0][:self.length]
            pred_nonrefclass_loc=torch.where(pos==0)[0][:self.length]
            pos_data = self.data_tensor[pred_refclass_loc]
            neg_data = self.data_tensor[pred_nonrefclass_loc]
            pos_styles, neg_styles = [], []
            for pos_data_batch, neg_data_batch in zip(DataLoader(pos_data,batch_size=5),DataLoader(neg_data,batch_size=5)):
                pos_data_batch = pos_data_batch.to(device)
                neg_data_batch = neg_data_batch.to(device)
                _ , pos_style = self.inverter(pos_data_batch.float(), randomize_noise=False, return_latents=True)
                _ , neg_style = self.inverter(neg_data_batch.float(), randomize_noise=False, return_latents=True)
                pos_styles.append(pos_style.detach().cpu())
                neg_styles.append(neg_style.detach().cpu())
                pos_data_batch.detach().cpu()
                neg_data_batch.detach().cpu()
            pos_styles = torch.concat(pos_styles, dim=0)
            neg_styles = torch.concat(neg_styles, dim=0)
            # print(pos_styles.shape,neg_styles.shape)
            pos_noise = torch.mean(pos_styles, dim=1)
            neg_noise = torch.mean(neg_styles, dim=1)
        else: 
            pos=converted_score[self.classno]
            pred_refclass_loc=torch.where(pos==1)[0][:self.length]
            pos_data = self.data_tensor[pred_refclass_loc]
            pos_styles = []
            for pos_data_batch in DataLoader(pos_data,batch_size=5):
                pos_data_batch = pos_data_batch.to(device)
                _ , pos_style = self.inverter(pos_data_batch.float(), randomize_noise=False, return_latents=True)
                pos_styles.append(pos_style.detach().cpu())
                pos_data_batch.detach().cpu()
            pos_styles = torch.cat(pos_styles, dim=0)
            pos_noise = torch.mean(pos_styles, dim=1)
            neg_noise = self.mine_neg_examples(pos_noise_tensor=pos_noise) 
        return  pos_noise, neg_noise

