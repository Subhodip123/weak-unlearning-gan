import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import AUROC
from torchvision.utils import save_image
from utils.dcgan import Generator
from utils.lenet import LeNet5
from utils.helper import no_ref_class, metrics, fid_density_coverage
from generator import DataGenerator
from utils.user_feedback import UserFeedback
from blockgan import MultirefBlockGAN, MultirefSVMBlockGAN
sys.path.insert(0, '/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist')
from model import BaseClassifier



# ---------Device settings-----------
warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
torch.manual_seed(46)



#-------------Generate Data---------
G = Generator(ngpu=1).eval()
G.requires_grad_(False)
# load weights
G.load_state_dict(torch.load("/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_gan/dcgan_checkpoints/mnist/netG_epoch_99.pth"))
if torch.cuda.is_available():
    G = G.cuda()
datagen = DataGenerator(pretrained_gan=G, no_samples=10000, batch_size = 100, img_dims=(1, 28, 28))
train_tensor, train_noise_tensor, test_tensor, test_noise_tensor = datagen.train_test_split()
train_tensor = train_tensor.to(device)
train_noise_tensor = train_noise_tensor.to(device)
test_tensor = test_tensor.to(device)
test_noise_tensor = test_noise_tensor.to(device)
print("No of generated data=",len(test_tensor))



#-----------Target Class Calculation------
org_classifiernet = LeNet5().eval()
org_classifiernet.load_state_dict(torch.load("/home/ece/Subhodip/Unlearning/UnlearnGAN/weights_classifier/mnist/lenet_epoch=12_test_acc=0.991.pth"))
org_classifiernet = org_classifiernet.to(device)
class_8_nos, org_labels = no_ref_class(net=org_classifiernet, gen_data=test_tensor, classno=8)
print("No class-8 in generated data=", class_8_nos)



#--------------Base Classifier Evaluation-------------
def baseclassifier_eval(length, data_tensor, class_no: int, actual_labels: torch.tensor, aug=False):
    actual_labels = actual_labels.to(device)
    # classifier = BaseClassifier(inchannels=1)
    # classifier = classifier.to(device=device)
    if aug==True:
        path = f'/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results/class-8/augclassifier{length}.pt'
    else:
        path = f'/home/ece/Subhodip/Unlearning/UnlearnGAN/Unlearn-Blocking/baseclassifier/mnist/results/class-8/classifier{length}.pt'
    classifier = torch.load(path)
    classifier_labels = []
    for data_batch in DataLoader(data_tensor, batch_size=64):
        data_batch = transforms.Resize((224,224))(data_batch).to(device)
        batch_labels = classifier.predict(data_batch)
        classifier_labels.append(batch_labels)
        data_batch.detach().cpu()
    classifier_labels = torch.cat(classifier_labels, dim=0)
    classifier_labels = classifier_labels.reshape((classifier_labels.size()[0],))
    auc = AUROC(task='binary')
    auc_score = auc(classifier_labels, actual_labels).cpu().item()
    blocked_output = test_tensor[torch.where(classifier_labels==1)[0]]
    unblocked_output = test_tensor[torch.where(classifier_labels==0)[0]]
    no_ref_blocked,_ = no_ref_class(net=org_classifiernet, gen_data=blocked_output, classno = class_no)
    no_ref_unblocked,_ = no_ref_class(net=org_classifiernet, gen_data=unblocked_output, classno = class_no)
    print("blocked, unblocked, ref blocked, ref unblocked", 
          len(blocked_output), len(unblocked_output), no_ref_blocked, no_ref_unblocked)
    fid, density, coverage = fid_density_coverage(input_tensor=test_tensor, input_labels=actual_labels, 
                                         unblocked_tensor=unblocked_output)
    acc, prec, recall, _, f1_score = metrics(no_blocked=len(blocked_output), no_unblocked=len(unblocked_output),
                                                  blocked_ref=no_ref_blocked, unblocked_ref=no_ref_unblocked)
    return acc, prec, recall, f1_score, round(auc_score,3), fid, density, coverage



def blocking_eval(user_pos_neg):
    lengths = np.arange(20,101,20).tolist()  
    method = ['BaseClassifier','MultirefBlock', 'MultirefSVMBlock'] * len(lengths)
    references = []
    accs = []
    precs = []
    recalls = []
    f1_scores =[]
    auc_scores = []
    fid_scores =[]
    densities = []
    coverages = []
    #--------------Multi User Ref and Block gans----------#
    for length in lengths:
        print("no of ref", length)
        references.extend([length]*3)
        user_feedback = UserFeedback(data_tensor=train_tensor, noise_tensor=train_noise_tensor,
                                     classifier=org_classifiernet, classno=8, length=length, pos_neg=user_pos_neg)
        
        multirefblocker = MultirefBlockGAN(user_feedback=user_feedback,
                                           test_tensor = test_tensor,
                                        test_noise_tensors=test_noise_tensor, 
                                        classifier_net=org_classifiernet,
                                        classification_func=no_ref_class, class_no=8)
        
        multisvmblocker = MultirefSVMBlockGAN(user_feedback=user_feedback,
                                        test_noise_tensors=test_noise_tensor, 
                                        test_tensor = test_tensor,
                                        classifier_net=org_classifiernet,
                                        classification_func=no_ref_class, class_no=8)
        
        
        #base classifier
        acc_base, prec_base, recall_base, f1_score_base, auc_score_base, fid_base, density_base, coverage_base = baseclassifier_eval(length=length,
                                                                                        data_tensor=test_tensor,
                                                                                        class_no=8, 
                                                                                        actual_labels=org_labels, aug=True)
        print("acc precision recall f1 auc density coverage", acc_base, prec_base, recall_base, f1_score_base, 
              round(auc_score_base,3), fid_base, density_base, coverage_base)
        
        #multiref
        no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_multiref = multirefblocker.output_perturbation(aug=True)
        #save unblocked images 
        # save_image(ub_imgs[:100], f'./results/class-5/multirun/multiref_unblocked{length}.png', nrow=10, normalize=True, value_range=(-1,1))
        print("Multiblocker output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        acc_multiref, prec_multiref, recall_multiref, _ ,f1_score_multiref = metrics(no_blocked=no_b_imgs, 
                                                                                     no_unblocked=len(ub_imgs),
                                                      blocked_ref=no_refb, unblocked_ref=no_refub)
        fid_multiref, density_multiref, coverage_multiref = fid_density_coverage(input_tensor=test_tensor, input_labels=org_labels, unblocked_tensor=ub_imgs)
        print("acc precision recall f1 auc density coverage", acc_multiref, prec_multiref, recall_multiref, 
              f1_score_multiref, auc_score_multiref, fid_multiref, density_multiref, coverage_multiref)

        #multiref svm
        no_b_imgs, ub_imgs, no_refb, no_refub, auc_score_svm = multisvmblocker.output_perturbation(aug=True)
        # save_image(ub_imgs[:100], f'./results/class-5/multirun/multisvm_unblocked{length}.png', nrow=10, normalize=True, value_range=(-1,1))
        print("Multiblocker SVM output", no_b_imgs, len(ub_imgs), no_refb, no_refub)
        acc_svm, prec_svm, recall_svm, _ ,f1_score_svm = metrics(no_blocked=no_b_imgs, no_unblocked=len(ub_imgs),
                                                      blocked_ref=no_refb, unblocked_ref=no_refub)
        fid_svm, density_svm, coverage_svm = fid_density_coverage(input_tensor=test_tensor, input_labels=org_labels, unblocked_tensor=ub_imgs)                               
        print("acc precision recall f1 auc density coverage", acc_svm, prec_svm, recall_svm, f1_score_svm, auc_score_svm, fid_svm,
              density_svm, coverage_svm)
        
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
        accs.extend([acc_base, acc_multiref, acc_svm])
        precs.extend([prec_base, prec_multiref, prec_svm])
        recalls.extend([recall_base, recall_multiref, recall_svm])
        f1_scores.extend([f1_score_base, f1_score_multiref, f1_score_svm])
        auc_scores.extend([auc_score_base, auc_score_multiref, auc_score_svm])
        fid_scores.extend([fid_base, fid_multiref, fid_svm])
        densities.extend([density_base, density_multiref, density_svm])
        coverages.extend([coverage_base, coverage_multiref, coverage_svm])

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
    results_dic = blocking_eval(user_pos_neg=False)
    results_df = pd.DataFrame(results_dic)
    results_df.to_csv('./results/class-8/experiments12.csv')
    

