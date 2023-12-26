import numpy as np
import torch
from torch import nn
from scipy import linalg
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pdrc import compute_prdc
from torchvision.models import resnet18
from utils.inception import InceptionV3
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(44)

def no_ref_class(net, gen_data, classno: int,classifier="None"):
    """Given the user class no finds out what no belong the user class reference no"""


    if len(gen_data) != 0:

        gen_dataset=DataLoader(gen_data,batch_size=10)
        gen_data_list=[]
        pred_refclass_tensor=[]
        no_images=0

        if(classifier=="Base Classifier"):
            for batches in gen_dataset:
                batches = transforms.Resize((224, 224))(batches)
                # print(gen_data.size())
                pred_labels = net(batches).cpu().detach().numpy()
                pred_class = np.argmax(pred_labels, axis=1)
                print(pred_class)
                pred_class_tensor = torch.tensor(data=pred_class)
                pred_refclass_tensor.append(torch.where(pred_class_tensor==classno, 1, 0))
                no_images += np.sum(pred_class == classno) 
            
        
        else:
            for batches in gen_dataset:
            # print(gen_data.shape)
                batches = transforms.Resize((218, 178))(batches)
                batches = batches.to(device)
                # print(gen_data.size())
                # pred_labels = self.classifier(gen_data).cpu().detach().numpy()
                score=net(batches).cpu().detach()
                batches.cpu().detach()
                converted_score=score.clone()
                converted_score[converted_score>=0]=1
                converted_score[converted_score<0]=0
                converted_score=converted_score.t()
                temp=converted_score[classno].numpy()
                pred_refclass_tensor.append(torch.where(converted_score[classno]==1,0,1))
                no_images+=np.sum(temp==1)
        concatenated_pred_refclass = torch.cat(pred_refclass_tensor, dim=0)
        print(concatenated_pred_refclass.shape)
        return no_images,concatenated_pred_refclass

    else:
            no_images=0
            pred_refclass_tensor = None
            return no_images, pred_refclass_tensor

    

        
def metrics(no_blocked: int, no_unblocked:int, blocked_ref: int , unblocked_ref:int):
    """This gives accuracy, False positive and False negetive in percentage terms"""
    true_pos = blocked_ref
    false_pos = no_blocked - blocked_ref
    true_neg = no_unblocked - unblocked_ref 
    false_neg = unblocked_ref 
    #accuracy
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    #precision
    if true_pos == 0 and false_pos == 0:
        precision = 0
    else:
        precision = true_pos /( true_pos + false_pos)
    #recall
    if true_pos == 0 and false_neg ==0:
        recall = 0
    else:
        recall = true_pos/(true_pos+false_neg)
    #false alarm
    if false_pos == 0 and true_neg ==0:
        false_alarm = 0
    else:
        false_alarm = false_pos/(false_pos+true_neg)
    # f1_Score
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall/(precision+recall)
    return round(accuracy,3), round(precision,3), round(recall,3), round(false_alarm,3), round(f1_score,3)



def load_patched_inception_v3():
    # inception = inception_v3(pretrained=True)
    # inception_feat = Inception3Feature()
    # inception_feat.load_state_dict(inception.state_dict())
    inception_feat = InceptionV3([3], normalize_input=False)

    return inception_feat


@torch.no_grad()
def extract_feature_from_samples(feature_extractor, data_tensor, batch_size=10):
    data_loader = DataLoader(dataset=data_tensor, batch_size=batch_size)
    features = []
    for data_batch in data_loader:
        data_batch = data_batch.to(device)
        print(feature_extractor(data_batch)[0].shape)
        batch_features = feature_extractor(data_batch)[0].view(data_batch.shape[0], -1)
        features.append(batch_features)
        data_batch.detach().cpu()
    features = torch.cat(features, 0)
    return features


def calc_fid(real_tensor, fake_tensor, eps=1e-6):
    sample_mean = np.mean(fake_tensor.numpy(), axis=0)
    sample_cov = np.cov(fake_tensor.numpy(), rowvar=False)
    real_mean = np.mean(real_tensor.numpy(),axis=0)
    real_cov = np.cov(real_tensor.numpy(), rowvar=False)
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid



def fid_density_coverage(input_tensor: torch.tensor, input_labels:torch.tensor, unblocked_tensor:torch.tensor, back_bone = 'inception'):
    """Calculates the density and coverage metrics from the given tensor"""
    if len(unblocked_tensor) !=0:
        actual_released_tensor = input_tensor[torch.where(input_labels==1)[0]][:len(unblocked_tensor)]
        fake_released_tensor = unblocked_tensor
        if actual_released_tensor.size()[1] != 3:
           actual_released_tensor = torch.concat([actual_released_tensor, actual_released_tensor, actual_released_tensor], dim=1)
           fake_released_tensor = torch.concat([fake_released_tensor, fake_released_tensor, fake_released_tensor], dim=1)
        if back_bone=='resnet18':
            feature_extractor = resnet18( weights = 'DEFAULT', progress=False)
            feature_extractor.to(device=device)
            actual_released_tensor = transforms.Resize((224,224))(actual_released_tensor)
            fake_released_tensor = transforms.Resize((224,224))(fake_released_tensor)
        elif back_bone=='inception':
            inception = load_patched_inception_v3()
            feature_extractor = nn.DataParallel(inception).eval().to(device)     
            # actual_released_tensor = transforms.Resize((299,299))(actual_released_tensor)
            # fake_released_tensor = transforms.Resize((299,299))(fake_released_tensor)

        actual_features = extract_feature_from_samples(feature_extractor=feature_extractor, data_tensor=actual_released_tensor).cpu()
        fake_features = extract_feature_from_samples(feature_extractor=feature_extractor, data_tensor=fake_released_tensor).cpu()
        fid = calc_fid(real_tensor=actual_features, fake_tensor=fake_features)
        density, coverage = compute_prdc(real_features=actual_features, fake_features=fake_features, nearest_k=3)
        actual_released_tensor.cpu().detach()
        fake_released_tensor.cpu().detach()
    else:
        fid, density, coverage = 'High', 0,0
    return fid, density , coverage
