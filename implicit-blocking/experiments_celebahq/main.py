import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import wandb
from torchmetrics import AUROC
from utils.pretrained_classifier import model_classifier
from utils.helper import no_ref_class, metrics, fid_density_coverage
from generator import DataGenerator
from utils.user_feedback import UserFeedback
from model import BaseClassifier
from blockgan import MultirefBlockGAN, MultirefSVMBlockGAN
sys.path.insert(0, '/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq')
from model import BaseClassifier



# ---------Device settings-----------
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
torch.manual_seed(44)




#-------------Generate Data-----------------#
generator_checkpoint = '/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_gan/stylegan2_checkpoints/CelebAHQ_checkpoint/370000.pt'
datagen = DataGenerator(checkpoiint=generator_checkpoint, no_samples=10000, output_size=256, latent_dim=512)
train_tensor, train_noise_tensor, test_tensor, test_noise_tensor = datagen.train_test_split()
print("No of generated data=",len(test_tensor),torch.max(train_tensor), torch.min(train_tensor))
#-------------------------------------------#



#--------------Target Class Calculation------#
#  percentage of data in having class 5
org_classifiernet = model_classifier(checkpoint_path='/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_classifier/celebahq/model_3_epoch.pt')
org_classifiernet = org_classifiernet.to(device)
class_5_nos, class_pred = no_ref_class(net=org_classifiernet, gen_data=test_tensor, classno=36, classifier="None")
print("No hats in generated data=", class_5_nos)
# print("location", in_class_location)
#--------------------------------------------#




##--------------Base Classifier Evaluation---------------##
def baseclassifier_eval(length, class_no: int, actual_labels: torch.tensor, aug=False):
    #-----------Base classifier-------------------#
    # actual_labels = actual_labels.to(device)
    # classifier = BaseClassifier(inchannels=1)
    # classifier = classifier.to(device=device)
    if aug == False:
        path = f'/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results/class-hats/classifier{length}.pt'
    else:
        path = f'/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/celebahq/results/class-hats/augclassifier{length}.pt'
    
    classifier = torch.load(path)
    test_dataset=DataLoader(test_tensor,batch_size=64)
    classifier_labels=[]
    for data_batch in test_dataset:
        data_batch = transforms.Resize((224,224))(data_batch).to(device)
        classifier_labels.append(classifier.predict(data_batch))
    classifier_labels=torch.cat(classifier_labels, dim=0)
    # print(test_tensor.shape)
    # print(classifier_labels.shape)
    # classifier_labels = classifier.predict(test_tensor)
    classifier_labels = classifier_labels.reshape((classifier_labels.size()[0],)).to("cpu")
    auc = AUROC(task='binary')
    auc_score = auc(classifier_labels, actual_labels).cpu().detach().item()
    blocked_output = test_tensor[torch.where(classifier_labels==1)[0]]
    unblocked_output = test_tensor[torch.where(classifier_labels==0)[0]]
    no_ref_blocked,_ = no_ref_class(net=org_classifiernet,gen_data=blocked_output, classno = class_no,classifier=None)
    no_ref_unblocked,_ = no_ref_class(net=org_classifiernet, gen_data=unblocked_output, classno = class_no,classifier=None)
    print("blocked, unblocked, ref blocked, ref unblocked", 
          len(blocked_output), len(unblocked_output), no_ref_blocked, no_ref_unblocked)
    fid, density, coverage = fid_density_coverage(input_tensor=test_tensor, input_labels=actual_labels, 
                                         unblocked_tensor=unblocked_output)
    
    acc, prec, recall, _, f1_score = metrics(no_blocked=len(blocked_output), no_unblocked=len(unblocked_output),
                                                  blocked_ref=no_ref_blocked, unblocked_ref=no_ref_unblocked)
    return acc, prec, recall, f1_score, round(auc_score,3), fid, density, coverage



def blocking_eval(user_pos_neg):
    #------------------set up-----------------#
    lengths = np.arange(20,61,20).tolist() #+ np.arange(200,1001,200).tolist() + [2000,3000,4000,5000]  
    method = ['MultirefBlock', 'MultirefSVMBlock'] * len(lengths)
    references = []
    accs = []
    precs = []
    recalls = []
    f1_scores =[]
    auc_scores = []
    fid_scores = []
    densities = []
    coverages = []
    #--------------Multi User Ref and Block gans----------#
    for length in lengths:
        print("no of ref", length)
        references.extend([length]*2)
        user_feedback = UserFeedback(data_tensor=train_tensor, noise_tensor=train_noise_tensor,
                                     classifier=org_classifiernet, classno=36, length=length,pos_neg=user_pos_neg)

        
        multirefblocker = MultirefBlockGAN(user_feedback=user_feedback,
                                           test_tensor = test_tensor,
                                        test_noise_tensors=test_noise_tensor, 
                                        classifier_net=org_classifiernet,
                                        classification_func=no_ref_class,
                                        class_no=36)
        multisvmblocker = MultirefSVMBlockGAN(user_feedback=user_feedback,
                                        test_noise_tensors=test_noise_tensor, 
                                        test_tensor = test_tensor,
                                        classifier_net=org_classifiernet,
                                        classification_func=no_ref_class,
                                        class_no=36)
        
        # multimlpblocker = MultirefMLPBlockGAN(user_feedback=user_feedback,
        #                                       latent_classifier_path=latent_classifier_path,
        #                                       test_tensor = test_tensor,
        #                                     test_noise_tensors=test_noise_tensor, 
        #                                     classifier_net=org_classifiernet,
        #                                     classification_func=no_ref_class)
        


        # #base classifier
        # acc_base, prec_base, recall_base, f1_score_base, auc_score_base, fid_base, density_base, coverage_base = baseclassifier_eval(length=length, 
        #                                                                                   class_no=36, 
        #                                                                                   actual_labels=class_pred, 
        #                                                                                   aug=True)
        # print("acc precision recall f1 auc fid density coverage", acc_base, prec_base, recall_base, f1_score_base, 
        #       round(auc_score_base,3), fid_base, density_base, coverage_base)
        


        #multiref
        no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_multiref = multirefblocker.output_perturbation(augment=True)
        # save_image(ub_imgs[:100], f'./results/class-bangs/multirun/multiref_unblocked{length}.png', nrow=10, normalize=True, value_range=(-1,1))
        print("Multiblocker output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        acc_multiref, prec_multiref, recall_multiref, _ ,f1_score_multiref = metrics(no_blocked=no_b_imgs, 
                                                                                     no_unblocked=len(ub_imgs),
                                                      blocked_ref=no_refb, unblocked_ref=no_refub)
        fid_multiref, density_multiref, coverage_multiref = fid_density_coverage(input_tensor=test_tensor, input_labels=class_pred, unblocked_tensor=ub_imgs)
        print("acc precision recall f1 auc fid density coverage", acc_multiref, prec_multiref, recall_multiref, 
              f1_score_multiref, auc_score_multiref, fid_multiref, density_multiref, coverage_multiref)

        
        
        #multiref svm
        no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_svm = multisvmblocker.output_perturbation(augment=True)
        # save_image(ub_imgs[:100], f'./results/class-bangs/multirun/multiref_unblocked{length}.png', nrow=10, normalize=True, value_range=(-1,1))
        print("Multiblocker SVM output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        acc_svm, prec_svm, recall_svm, _ ,f1_score_svm = metrics(no_blocked=no_b_imgs, no_unblocked=len(ub_imgs),
                                                      blocked_ref=no_refb, unblocked_ref=no_refub)
        fid_svm, density_svm, coverage_svm = fid_density_coverage(input_tensor=test_tensor, input_labels=class_pred, unblocked_tensor=ub_imgs)                               
        print("acc precision recall f1 auc density coverage", acc_svm, prec_svm, recall_svm, f1_score_svm, auc_score_svm, 
              fid_svm, density_svm, coverage_svm)
        
        # #multiref mlp
        # no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_mlp = multimlpblocker.output_perturbation()
        # print("Multiblocker mlp output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        # acc_mlp, prec_mlp, recall_mlp, _ ,f1_score_mlp = metrics(no_blocked=no_b_imgs, no_unblocked=len(ub_imgs),
        #                                               blocked_ref=no_refb, unblocked_ref=no_refub)
        # density_mlp, coverage_mlp = fid_density_coverage(input_tensor=test_tensor, input_labels=class_pred, unblocked_tensor=ub_imgs)
        # print("acc precision recall f1 auc density coverage", acc_mlp, prec_mlp, recall_mlp, f1_score_mlp, auc_score_mlp,
        #       density_mlp, coverage_mlp)
        
        # #multiref encoder
        # no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_enc = encoderblocker.output_perturbation()
        # print("Multiblocker encoder output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        # acc_enc, prec_enc, recall_enc, _ ,f1_score_enc = metrics(no_blocked=no_b_imgs, no_unblocked=len(ub_imgs),
        #                                               blocked_ref=no_refb, unblocked_ref=no_refub)
        # print("acc precision recall f1 auc", acc_enc, prec_enc, recall_enc, f1_score_enc, auc_score_enc)

        #update
        accs.extend([acc_multiref, acc_svm])
        precs.extend([prec_multiref, prec_svm])
        recalls.extend([recall_multiref, recall_svm])
        f1_scores.extend([ f1_score_multiref, f1_score_svm])
        auc_scores.extend([auc_score_multiref, auc_score_svm])
        fid_scores.extend([fid_multiref, fid_svm])
        densities.extend([density_multiref, density_svm])
        coverages.extend([coverage_multiref, coverage_svm])

    results_dic = {'methods': method,
                   'pos_neg_ref': references,
                   'accuracy': accs,
                   'precision':precs,
                   'recall': recalls,
                   'f1 scores':f1_scores,
                   'AUC': auc_scores,
                   'FID' : fid_scores,
                   'densities': densities,
                   'coverages': coverages}

    return results_dic
    #---------------------------------------------------#



if __name__ == '__main__':

    
    #-----------evaluation----------------#
    results_dic = blocking_eval(user_pos_neg=True)
    results_df = pd.DataFrame(results_dic)
    results_df.to_csv('./results/class-hats/experiments7.csv')
    

