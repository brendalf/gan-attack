# Changelog

Here I will be writting the articles that I've read and a description about it.

## 30.03.20

#### Sentiment Analysis with Contextual Embeddings and Self-Attention
Authors: Katarzyna Biesialska, Magdalena Biesialska and Henryk Rybinski.  
Release Date: 12.03.2020

The paper presents the Transformer-based Sentiment Analysis (TSA), a multilingual sentiment 
classifier model that is comparable or even outperforms state-of-the-art models for languages like 
Polish and German. The model is based on the Transformer architecture with a attention-based layer.

## 19.07.2020

#### Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data
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

## 21.07.2020

#### Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser
Authors: Fangzhou Liao, Ming Liang, Yinpeng Dong, Tianyu Pang, Xiaolin Hu, Jun Zhu
Release Date: 08.05.2018

The paper proposes a autoencoder network called High-Level Representation Guided Denoiser (HGD) as 
a defense for image classification. This network acts as a defender for a target network. It 
generalizes well to other unseen classes and transferability to other target networks.

The autoencoder proposed is a modified version of DAE (denoiser autoencoder) with U-net structure. 
It has lateral connections from encoder layers to their corresponding decoder layers and the 
learning objective is the adversarial noise.

## 24.07.2020

#### Stealing Knowledge from Protected Deep Neural Networks Using Composite Unlabeled Data
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

## 25.07.2020

#### DAWN: Dynamic Adversarial Watermarking of Neural Networks
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