import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.helper import metrics
from sklearn.svm import SVC
from sklearn.metrics import auc
from torch.distributions.normal import Normal



class MultirefBlockGAN(object):
    """blocks the output of the user_ref image"""
    def __init__(self, user_feedback, test_tensor, test_noise_tensors, classifier_net, classification_func, class_no) -> None:
        self.user_feedback = user_feedback
        self.test_tensors = test_tensor
        self.test_noise_tensors = test_noise_tensors
        self.classifier_net = classifier_net
        self.classification_func = classification_func
        self.class_no = class_no
       

    def target_pos_neg_latent(self, aug=False):
        """It returns the target class's representation"""

        z_pos, z_neg= self.user_feedback.feedback_tensorization()

        if aug == True:
            pos_mean = torch.mean(z_pos, dim=0)
            pos_std = torch.std(z_pos, dim=0)
            neg_mean = torch.mean(z_neg, dim=0)
            neg_std = torch.std(z_neg, dim=0)
            pos_dist = Normal(pos_mean, pos_std)
            neg_dist = Normal(neg_mean, neg_std)
            pos_sam = pos_dist.sample((5000,))
            neg_sam = neg_dist.sample((5000,))
            # print('Hi',pos_sam.shape)
            z_pos = torch.concat([z_pos,pos_sam], dim=0)
            z_neg = torch.concat([z_neg,neg_sam], dim=0)

        z_target = torch.mean(z_pos, dim=0) - torch.mean(z_neg, dim=0)
        return z_target, z_pos, z_neg
    
    def projection_similarity(self, z_target: torch.tensor, z_input: torch.tensor):
        """calculates the cosine similarity between the reference img and input"""
        
        similarity_arr = torch.zeros(size=(z_input.size()[0],))
        batch_size = z_input.size()[0]
        for i in range(batch_size):
            similarity_score = torch.dot(z_input[i, :], z_target) / torch.norm(z_target) 
            similarity_arr[i] = similarity_score 
        return similarity_arr

    def output_perturbation(self, aug=False):
        """It will give purturbed output based on the learning"""
        z_target, z_pos, z_neg = self.target_pos_neg_latent(aug=aug)
        z_test = self.test_noise_tensors
        pos_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_pos)
        neg_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_neg)
        test_similarity_arr = self.projection_similarity(z_target=z_target, z_input=z_test)
        tau_max = torch.max(test_similarity_arr).item()
        tau_min = torch.min(test_similarity_arr).item()
        pos_tau_sum = torch.sum(pos_sim_arr).item()
        neg_tau_sum = torch.sum(neg_sim_arr).item()
        threshold_tau = (pos_tau_sum+neg_tau_sum)/(len(pos_sim_arr)+len(neg_sim_arr))
        step = (tau_max-tau_min) / 1000
        print(step)
        print(tau_max)
        print(tau_min)
        print("Mean thereshold value=", threshold_tau)
        tau_range = np.arange(tau_min, tau_max, step)
        no_ref_blocked = []
        no_blocked_imgs = []
        detection_prob = []
        false_alarm_prob = []
        total_ref_img,_ = self.classification_func(net=self.classifier_net, gen_data = self.test_tensors, classno=self.class_no)
        # print(total_ref_img)
        for tau in np.arange(tau_min, tau_max, step).tolist():
            unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= tau)[0]]
            unblocked_ref  ,_  = self.classification_func(net=self.classifier_net, 
                                                              gen_data=unblocked_output, classno=self.class_no)
            blocked_output = len(self.test_tensors)-len(unblocked_output)
            blocked_ref = total_ref_img - unblocked_ref
            _,_, recall, false_alarm ,_ = metrics(no_blocked=blocked_output, no_unblocked= len(unblocked_output),
                                                  blocked_ref=blocked_ref, unblocked_ref=unblocked_ref)
            detection_prob.append(recall)
            false_alarm_prob.append(false_alarm) 
            no_blocked_imgs.append(blocked_output)
            no_ref_blocked.append(blocked_ref)

        #blocking at threshold tau
        threshold_unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= threshold_tau)[0]]
        threshold_blocked_imgs = len(self.test_tensors)-len(threshold_unblocked_output)
        threshold_ref_unblocked ,_  = self.classification_func(net=self.classifier_net, 
                                                                 gen_data=threshold_unblocked_output, classno=self.class_no)
        threshold_ref_blocked = total_ref_img - threshold_ref_unblocked
        plt.figure(figsize=(10, 10))
        plt.plot(
            tau_range,
            no_ref_blocked,
            linewidth="3",
            color="k",
            alpha=0.3,
        )
        plt.plot(tau_range,
                 no_blocked_imgs,
                 linewidth='2',
                 color='b',)
        plt.axvline(x=threshold_tau, color='r')
        # plt.axvline(x=neg_tau_mean, color='g')
        plt.legend([f"no of ref-class in blocked output={threshold_blocked_imgs}", 
                    f"no of blocked outputs={threshold_ref_blocked}"])

        plt.title("blocking results vs tau")
        fig = plt.gcf()
        fig.savefig(f'./results/class-8/multirun/multiref_blocking{len(z_pos)}.pdf')

        #ROC curve
        auc_score = round(auc(false_alarm_prob, detection_prob),3)
        plt.figure(figsize=(10, 10))
        plt.plot(
            false_alarm_prob,
            detection_prob,
            linewidth="3",
            color="k",
            alpha=0.3,
        )
        plt.title("ROC curve")
        plt.legend([f"auc score={auc_score}"])
        fig2 = plt.gcf()
        fig2.savefig(f'./results/class-8/multirun/roc_multiref{len(z_pos)}.pdf')
        return threshold_blocked_imgs, threshold_unblocked_output, threshold_ref_blocked, threshold_ref_unblocked, auc_score
    



class MultirefSVMBlockGAN(object):
    """blocks the output of the user_ref image"""
    def __init__(self, user_feedback, test_tensor, test_noise_tensors, classifier_net, classification_func, class_no) -> None:
        self.user_feedback = user_feedback
        self.test_tensors = test_tensor
        self.test_noise_tensors = test_noise_tensors
        self.classifier_net = classifier_net
        self.classification_func = classification_func
        self.class_no = class_no
        # print("no of ref",len(self.pos_ref_imgs))

    def classify(self, z_pos: torch.tensor, z_neg: torch.tensor):
        """Given positive and negetive latents gives the normal of the hyperplane"""
    
        classifier = SVC(kernel="linear", max_iter=1000)
        pos_data = z_pos.cpu().detach().numpy()
        neg_data = z_neg.cpu().detach().numpy()
        pos_label = np.ones((pos_data.shape)[0])
        neg_label = np.ones((neg_data.shape)[0]) * -1
        labels = np.concatenate([pos_label, neg_label], axis=0)
        # print("labels", labels.shape)
        data = np.concatenate([pos_data, neg_data], axis=0)
        # print("data", data.shape)
        classifier.fit(data, labels)
        normal_tensor = torch.tensor(data=classifier.coef_).cuda()
        normal_tensor = normal_tensor.reshape((normal_tensor.shape[1],)).type(torch.float32)
        return normal_tensor
    
    def target_pos_neg_latent(self, aug=False):
        """It returns the target class's representation"""

        z_pos, z_neg= self.user_feedback.feedback_tensorization()

        if aug == True:
            pos_mean = torch.mean(z_pos, dim=0)
            pos_std = torch.std(z_pos, dim=0)
            neg_mean = torch.mean(z_neg, dim=0)
            neg_std = torch.std(z_neg, dim=0)
            pos_dist = Normal(pos_mean, pos_std)
            neg_dist = Normal(neg_mean, neg_std)
            pos_sam = pos_dist.sample((5000,))
            neg_sam = neg_dist.sample((5000,))
            z_pos = torch.concat([z_pos,pos_sam], dim=0)
            z_neg = torch.concat([z_neg,neg_sam], dim=0)
        z_target = self.classify(z_pos=z_pos, z_neg=z_neg)
        return z_target, z_pos, z_neg
    
    def projection_similarity(self, z_target: torch.tensor, z_input: torch.tensor):
        """calculates the cosine similarity between the reference img and input"""
        
        similarity_arr = torch.zeros(size=(z_input.size()[0],))
        batch_size = z_input.size()[0]
        for i in range(batch_size):
            similarity_score = torch.dot(z_input[i, :], z_target) / torch.norm(z_target) 
            similarity_arr[i] = similarity_score 
        return similarity_arr

    def output_perturbation(self, aug=False):
        """It will give purturbed output based on the learning"""
        z_target, z_pos, z_neg = self.target_pos_neg_latent(aug=aug)
        z_test = self.test_noise_tensors
        pos_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_pos)
        neg_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_neg)
        test_similarity_arr = self.projection_similarity(z_target=z_target, z_input=z_test)
        tau_max = torch.max(test_similarity_arr).item()
        tau_min = torch.min(test_similarity_arr).item()
        pos_tau_sum = torch.sum(pos_sim_arr).item()
        neg_tau_sum = torch.sum(neg_sim_arr).item()
        threshold_tau = (pos_tau_sum+neg_tau_sum)/(len(pos_sim_arr)+len(neg_sim_arr))
        step = (tau_max-tau_min)/ 1000
        print(tau_max)
        print(tau_min)
        print("Mean thereshold value=", threshold_tau)
        tau_range = np.arange(tau_min, tau_max, step)
        no_ref_blocked = []
        no_blocked_imgs = []
        detection_prob = []
        false_alarm_prob = []
        total_ref_img,_ = self.classification_func(net=self.classifier_net, gen_data = self.test_tensors, classno=self.class_no)
        # print(total_ref_img)
        for tau in np.arange(tau_min, tau_max, step).tolist():
            unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= tau)[0]]
            unblocked_ref  ,_ = self.classification_func(net=self.classifier_net, 
                                                              gen_data=unblocked_output, classno=self.class_no)
            blocked_output = len(self.test_tensors)-len(unblocked_output)
            blocked_ref = total_ref_img - unblocked_ref
            _,_, recall, false_alarm ,_ = metrics(no_blocked=blocked_output, no_unblocked= len(unblocked_output),
                                                  blocked_ref=blocked_ref, unblocked_ref=unblocked_ref)
            detection_prob.append(recall)
            false_alarm_prob.append(false_alarm) 
            no_blocked_imgs.append(blocked_output)
            no_ref_blocked.append(blocked_ref)

        #blocking at threshold tau
        threshold_unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= threshold_tau)[0]]
        threshold_blocked_imgs = len(self.test_tensors)-len(threshold_unblocked_output)
        threshold_ref_unblocked ,_  = self.classification_func(net=self.classifier_net, 
                                                                 gen_data=threshold_unblocked_output, classno=self.class_no)
        threshold_ref_blocked = total_ref_img - threshold_ref_unblocked
        plt.figure(figsize=(10, 10))
        plt.plot(
            tau_range,
            no_ref_blocked,
            linewidth="3",
            color="k",
            alpha=0.3,
        )
        plt.plot(tau_range,
                 no_blocked_imgs,
                 linewidth='2',
                 color='b',)
        plt.axvline(x=threshold_tau, color='r')
        # plt.axvline(x=neg_tau_mean, color='g')
        plt.legend([f"no of ref-class in blocked output={threshold_blocked_imgs}", 
                    f"no of blocked outputs={threshold_ref_blocked}"])

        plt.title("blocking results vs tau")
        fig = plt.gcf()
        fig.savefig(f'./results/class-8/multirun/multirefsvmblocking{len(z_pos)}.pdf')
        #ROC curve
        auc_score = round(auc(false_alarm_prob, detection_prob),3)
        plt.figure(figsize=(10, 10))
        plt.plot(
            false_alarm_prob,
            detection_prob,
            linewidth="3",
            color="k",
            alpha=0.3,
        )
        plt.title("ROC curve")
        plt.legend([f"auc score={auc_score}"])
        fig2 = plt.gcf()
        fig2.savefig(f'./results/class-8/multirun/roc_svm{len(z_pos)}.pdf')
        return threshold_blocked_imgs, threshold_unblocked_output, threshold_ref_blocked, threshold_ref_unblocked, auc_score
    
    
# class MultirefMLPBlockGAN(object):
#     """blocks the output of the user_ref image"""
#     def __init__(self, user_feedback, test_tensor, test_noise_tensors, latent_classifier_path,
#                  classifier_net, classification_func, class_no ) -> None:
#         self.user_feedback = user_feedback
#         self.test_tensors = test_tensor.to(device=device)
#         self.test_noise_tensors = test_noise_tensors.to(device=device)
#         self.latent_classifier = torch.load(latent_classifier_path)
#         self.classifier_net = classifier_net
#         self.classification_func = classification_func
#         self.class_no = class_no
#         # print("no of ref",len(self.pos_ref_imgs))
    
#     def target_pos_neg_latent(self):
#         """It returns the target class's representation"""
#         z_pos , z_neg = self.user_feedback.feedback_tensorization()
#         normal_vec = self.latent_classifier.last_layer.weight
#         z_target = normal_vec.reshape((normal_vec.size()[1],))
#         return z_target, z_pos, z_neg
    
#     def projection_similarity(self, z_target: torch.tensor, z_input: torch.tensor):
#         """calculates the cosine similarity between the reference img and input"""
        
#         similarity_arr = torch.zeros(size=(z_input.size()[0],))
#         batch_size = z_input.size()[0]
#         for i in range(batch_size):
#             similarity_score = torch.dot(z_target, z_input[i, :]) / torch.norm(z_target) 
#             similarity_arr[i] = similarity_score 
#         # best_sim_score = torch.max(torch.abs(similarity_arr))
#         # similarity_arr = similarity_arr / best_sim_score
#         return similarity_arr

#     def output_perturbation(self):
#         """It will give purturbed output based on the learning"""
#         z_target, z_pos, z_neg = self.target_pos_neg_latent()
#         z_test = self.test_noise_tensors
#         pos_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_pos)
#         neg_sim_arr = self.projection_similarity(z_target=z_target, z_input=z_neg)
#         test_similarity_arr = self.projection_similarity(z_target=z_target, z_input=z_test)
#         # best_sim_score = max([torch.max(torch.abs(pos_sim_arr)), 
#         #                           torch.max(torch.abs(neg_sim_arr)),
#         #                           torch.max(torch.abs(test_similarity_arr))])
#         # pos_sim_arr = pos_sim_arr/ best_sim_score
#         # neg_sim_arr = neg_sim_arr / best_sim_score
#         # test_similarity_arr = test_similarity_arr/ best_sim_score
#         tau_max = torch.max(test_similarity_arr).item()
#         tau_min = torch.min(test_similarity_arr).item()
#         pos_tau_sum = torch.sum(pos_sim_arr).item()
#         neg_tau_sum = torch.sum(neg_sim_arr).item()
#         threshold_tau = (pos_tau_sum+neg_tau_sum)/(len(pos_sim_arr)+len(neg_sim_arr))
#         step = (tau_max-tau_min) / 1000
#         print(tau_max)
#         print(tau_min)
#         print("Mean thereshold value=", threshold_tau)
#         tau_range = np.arange(tau_min, tau_max, step)
#         no_ref_blocked = []
#         no_blocked_imgs = []
#         detection_prob = []
#         false_alarm_prob = []
#         total_ref_img,_ = self.classification_func(net=self.classifier_net, gen_data = self.test_tensors, classno=self.class_no)
#         # print(total_ref_img)
#         for tau in np.arange(tau_min, tau_max, step).tolist():
#             unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= tau)[0]]
#             unblocked_ref  ,_ = self.classification_func(net=self.classifier_net, 
#                                                               gen_data=unblocked_output, classno=self.class_no)
#             blocked_output = len(self.test_tensors)-len(unblocked_output)
#             blocked_ref = total_ref_img - unblocked_ref
#             _,_, recall, false_alarm ,_ = metrics(no_blocked=blocked_output, no_unblocked= len(unblocked_output),
#                                                   blocked_ref=blocked_ref, unblocked_ref=unblocked_ref)
#             detection_prob.append(recall)
#             false_alarm_prob.append(false_alarm) 
#             no_blocked_imgs.append(blocked_output)
#             no_ref_blocked.append(blocked_ref)

#         #blocking at threshold tau
#         threshold_unblocked_output = self.test_tensors[torch.where(test_similarity_arr <= threshold_tau)[0]]
#         threshold_blocked_imgs = len(self.test_tensors)-len(threshold_unblocked_output)
#         threshold_ref_unblocked ,_ = self.classification_func(net=self.classifier_net, 
#                                                                  gen_data=threshold_unblocked_output, classno=self.class_no)
#         threshold_ref_blocked = total_ref_img - threshold_ref_unblocked
#         plt.figure(figsize=(10, 10))
#         plt.plot(
#             tau_range,
#             no_ref_blocked,
#             linewidth="3",
#             color="k",
#             alpha=0.3,
#         )
#         plt.plot(tau_range,
#                  no_blocked_imgs,
#                  linewidth='2',
#                  color='b',)
#         plt.axvline(x=threshold_tau, color='r')
#         # plt.axvline(x=neg_tau_mean, color='g')
#         plt.legend([f"no of ref-class in blocked output={threshold_blocked_imgs}", 
#                     f"no of blocked outputs={threshold_ref_blocked}"])

#         plt.title("blocking results vs tau")
#         fig = plt.gcf()
#         fig.savefig(f'./results/class-8/multirun/multirefmlpblocking{len(z_pos)}.pdf')
#         #ROC curve
#         auc_score = round(auc(false_alarm_prob, detection_prob),3)
#         plt.figure(figsize=(10, 10))
#         plt.plot(
#             false_alarm_prob,
#             detection_prob,
#             linewidth="3",
#             color="k",
#             alpha=0.3,
#         )
#         plt.title("ROC curve")
#         plt.legend([f"auc score={auc_score}"])
#         fig2 = plt.gcf()
#         fig2.savefig(f'./results/class-8/multirun/roc_mlp{len(z_pos)}.pdf')
#         return threshold_blocked_imgs, threshold_unblocked_output, threshold_ref_blocked, threshold_ref_unblocked, auc_score
    
    
