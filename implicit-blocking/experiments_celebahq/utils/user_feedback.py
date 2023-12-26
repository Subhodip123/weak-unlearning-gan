import os
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class UserFeedback(object):
    """Reference class for User feedback"""

    def __init__(self, data_tensor, noise_tensor, classifier, classno, length, pos_neg=True) -> None:
        self.data_tensor = data_tensor
        self.noise_tensor = noise_tensor
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
                similarity_arr[i] = similarity_score 
            return similarity_arr
        

    def mine_neg_examples(self, pos_noise_tensor:torch.tensor) -> torch.tensor:
        """Given the inverter, generated tensor, and positive user"""
        z_pos = pos_noise_tensor
        z_gen = self.noise_tensor
        z_pos_mean = torch.mean(z_pos, dim=0)
        pos_sim_arr = self.projection_similarity(z_pos_mean, z_pos)
        generated_sim_arr = self.projection_similarity(z_pos_mean, z_gen)
        neg_loc = torch.where(generated_sim_arr<torch.min(pos_sim_arr)/2)[0]
        random_selected_locations = np.random.choice(neg_loc.tolist(), size=len(z_pos))
        gen_neg_tensor = self.noise_tensor[torch.tensor(random_selected_locations)]
        return gen_neg_tensor
    

    def feedback_tensorization(self):
        """given the directory of user ref images it reads all the images into a tensor"""
        # print(self.data_tensor.shape,self.noise_tensor.shape)
        gen_data = transforms.Resize((218, 178))(self.data_tensor)
        # print(gen_data.size())
        gen_dataloader=DataLoader(gen_data,batch_size=64)
        score=[]
        for data in gen_dataloader:
            data = data.to(device=device)
            # pred_labels = self.classifier(gen_data).cpu().detach().numpy()
            score.append(self.classifier(data).cpu().detach())
            data.cpu().detach()

        score=torch.cat(score,dim=0)
        # print(gen_data.shape,score.shape)
        converted_score=score.clone()
        converted_score[converted_score>=0]=1
        converted_score[converted_score<0]=0
        converted_score=converted_score.t()
        # pred_class = np.argmax(pred_labels, axis=1)
        # pred_class_tensor = torch.tensor(data=pred_class)
        if self.pos_neg:
            pos=converted_score[self.classno]
            pred_refclass_loc=torch.where(pos==1)[0][:self.length]
            pred_nonrefclass_loc=torch.where(pos==0)[0][:self.length]
            # pred_refclass_loc = torch.where(pred_class_tensor==self.classno)[0][:self.length]
            # pred_nonrefclass_loc = torch.where(pred_class_tensor!=self.classno)[0][:self.length]
            pos_noise = self.noise_tensor[pred_refclass_loc]
            neg_noise = self.noise_tensor[pred_nonrefclass_loc]    
        else: 
            # pred_refclass_loc = torch.where(pred_class_tensor==self.classno)[0][:self.length]
            pos=converted_score[self.classno]
            pred_refclass_loc=torch.where(pos==1)[0][:self.length]
            # print(pred_refclass_loc.shape)
            # print(self.noise_tensor.shape)
            pos_noise = self.noise_tensor[pred_refclass_loc]
            # print(pos_noise.shape)
            neg_noise = self.mine_neg_examples(pos_noise_tensor=pos_noise)
        
        print(pos_noise.shape,neg_noise.shape)
        return pos_noise, neg_noise

