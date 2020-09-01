# Changelog

Here I will be writting the articles that I've read and a description about it.

## Sentiment Analysis with Contextual Embeddings and Self-Attention
Authors: Katarzyna Biesialska, Magdalena Biesialska and Henryk Rybinski.  
Release Date: 12.03.2020

The paper presents the Transformer-based Sentiment Analysis (TSA), a multilingual sentiment 
classifier model that is comparable or even outperforms state-of-the-art models for languages like 
Polish and German. The model is based on the Transformer architecture with a attention-based layer.

## Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data
Authors: Jacson Rodrigues, Rodrigo F. Berriel, ..., Alberto F. de Souza, Thiago Oliveira-Santos
Release Date: 14.06.2018

The paper investigates if a target black-box CNN can be copied by persuading it to confess its 
knowledge through random non-labeled data. The hyphoteshis was evaluated in three problems: facial 
expression, object and crosswalk classification. One attack against a cloud-based API was also 
performed. 

Three types of attacks were performed: 1. non-problem domain data with stolen labels (passing 
images through the target cnn), 2. problem domain data with stolen labels and 3. training on 1 and 
finetunning on 2.

All copycat networks achieved at least 93.7% of performance of the original models with non-problem 
domain data and at least 98.6% using additional data from the problem domain.

## Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser
Authors: Fangzhou Liao, Ming Liang, Yinpeng Dong, Tianyu Pang, Xiaolin Hu, Jun Zhu
Release Date: 08.05.2018

The paper proposes a autoencoder network called High-Level Representation Guided Denoiser (HGD) as 
a defense for image classification. This network acts as a defender for a target network. It 
generalizes well to other unseen classes and transferability to other target networks.

The autoencoder proposed is a modified version of DAE (denoiser autoencoder) with U-net structure. 
It has lateral connections from encoder layers to their corresponding decoder layers and the 
learning objective is the adversarial noise.


## Stealing Knowledge from Protected Deep Neural Networks Using Composite Unlabeled Data
Authors: Itay Mosaf, Eli David, Nathan S. Netanyahu
Release Date: 19.07.2019

The paper proposes a Deep Neural Network adapted from VGG net that is able to steal knowledge
from others networks using the minimiun information required, the predicted class from 
the target model. In other words, the method proposed is able to steal knowledge even if the target
class doesn't return the softmax distribution of labels.

To train the student network, it's used the ImageNet with a augmentation algorithm that generates
one milion images from each epoch of training. Different every time. That's why other augmentation
methods are not applied. The generated images are a composition of two images from two different
classes.

The method proposed is able to mimic 99% performance from the target network and watermarking
defense aren't detectable.


## DAWN: Dynamic Adversarial Watermarking of Neural Networks
Authors: Sebastian Szyller, Buse Gul Atli, Samuel Marchal, N. Asokan
Release Date: 18.06.2020

The paper proposes a new defense system against deep neural networks stealing. It's the first approach to use watermaking to deter model extraction intellectual property theft and operates at the prediction API of the protected model, so doesn't impose changes to the training process. 

DAWN dynamically changes the responses for a small subset of queries from API clientes. This changes incurs a negligible loss of prediction accuracy while allow the model owners to reliably demonstrate ownership with high confidence.

The subset of queries with changed label by DAWN will be embedded inside the attacker model when the training occurs.

DAWN defines a function W that is resposible for decide if a input needs to be watermarked or not. DAWN also defines a mappping M function to clear images of small pertubations. This is necessary because an image x needs to get the same incorrect answer that x + s (small pertubation).

The papper doesn't try different approaches to this map function, but the map function proposed was tested for different levels of input pertubations and was stable.

DAWN is deterministic, so for the same query, it returns the same label, so the the attacker isn't abble to distinguish the correct from the watermarked labels by passing the dataset again through the target.

DAWN publishes inside a blockchain network sets of encoded images and encoded incorrect awnsers and the process to claim the authorship is by using a judge that pass the images through the suspect and see if the answers match the encoded incorrect ansers with some level of accuracy.

PROBLEMS:
* Double-extraction and fine-tunning can remove the watermark while preserving test acuucary, but the attacker should have unlimited access to natural data.
* The target model owner only can claim the to be the author if the stolen model exposes itself in the internet with an API. If the stolen model just be used inside a coorporation, the target model was never know. 
* The blockchain gets bigger and bigger and it presumes that the judge can query the suspect without any fee.
* DAWN should be tested against standard augmentation and composite augmentation.
* Use denoise autoencoder in replace to the traditional M function should get better results.

## Knockoff Nets: Stealing Functionality of Black-Box Models
Authors: Tribhuvanesh Orekondy, Bernt Schiele, Mario Fritz
Release Date: 06.12.2018

The papper proposes a method to steal functionality of black box models. It's kind of the same approach of the copycat attack. The main difference is that the output of the target network used to train the attacker is the softmax distribution, not just the argmax.

The method was evaluated for different network architectures of target models and the attacker was abble to steal a higher relative accuracy. The papper also tested two different sampling approaches: random and adaptive.

While performing the attack, the ImageNet was used. In the random sample strategy, the images are queried from ImageNet randomically and in the adaptive strategy, the model tries to identify the best images to steal functionality of the target model.

The adaptive strategy was abble to steal the same amount of information that the random sampling, but with less images.

## Data-Free Learning of Student Networks
Authors: Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi, Chunjing Xu, Chao Xu, Qi Tian
Release Date: 31.12.2019

The papper proposes a novel for training student networks with GAN.
Student network are understandable as more compact networks and the goal is to enable a network to be deployed in edge devices.

The Data-Free Learning (DAFL) was abble to achieve 92.22% and 74.47% of accuracy using ResNet-18 without any training data on the CIFAR-10 and CIFAR-100 datasets. Winning against other state-of-the-art methods.

One problem with this approach is that it uses the target model (teacher) as discriminator. So, applying this method for stealing knowledge of a network through a API is unavailable because of the number of API calls that will be necessary to train the generator.

The DAFL generator can effectively generate synthetic data that reproduces the original training set distribution.

Because the original dataset aren't available, training the discriminator to distinguish between real or fake from the original and the generator datasets is not posible. That's why the DALF uses a pre trained network as a fixed discriminator.

The student network uses the argmax of the teacher softmax output.

The generator loss is composed as three other losses:
* One hot encode loss (argmax of softmax)
* Feature maps activation loss (unavailable for api cases)
* Information entropy loss (to ensure balanced class datasets)