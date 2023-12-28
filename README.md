# FAST: Feature Aware Similarity Thresholding for Weak Unlearning in Black-Box Generative Models

This repo contains a PyTorch implementation for the paper [FAST: Feature Aware Similarity Thresholding for Weak Unlearning in Black-Box Generative Models](http://arxiv.org/abs/2312.14895)

--------------------

Modern machine unlearning approaches typically assume access to model parameters and architectural details during unlearning, which is not always feasible. In multitude of downstream tasks, these models functionas black-box systems, with inaccessible pre-trained parameters, architectures, and training data. In such scenarios, the possibility of filtering undesired outputs becomes a practical alternative. The primary goal of this study is twofold: first, to elucidate the relationship between filtering and unlearning processes, and second, to formulate a methodology aimed at mitigating the display of undesirable outputs generated from models characterized as black-box systems.  Theoretical analysis in this study demonstrates that, in the context of black-box models, filtering can be seen as a form of weak unlearning. Our proposed Feature Aware Similarity Thresholding(FAST) method effectively suppresses undesired outputs by systematically encoding the representation of unwanted features in the latent space.

![mechanism](blocking.png)


The primary objective of filtering is to prevent the display of samples that exhibit specific undesired features. In this context, we explore two distinct unlearning settings:

- **Class-level Filtering:** For this setting, we utilize the MNIST dataset (LeCun et al., 1998), which comprises 60,000 28 × 28 black and white images of handwritten digits. Aiming to achieve class-level filtering,  we have used pre-trained DC-GAN on MNIST. Specifically, we focus on filtering out two digit classes: 5 and 8.

- **Feature-level Filtering:** In this scenario, we turn to the CelebA-HQ dataset (Liu et al., 2015), which contains 30,000 high-quality RGB celebrity face images with dimensions of 256×256. For this, we have taken state-of-the-art StyleGAN2 pre-trained of CelebA-HQ. Here, we target the feature-level filtering of subtle features, specifically (a) Bangs and (b) Hats.


We provide evaluations of different filtering methods on the MNIST and CelebA-HQ datasets, respectively.We observe that our proposed methods give better Recall and AUC values for both datasets while comparable to superior performance for other metrics.