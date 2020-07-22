# Changelog

Here I will be writting the articles that I've read and a description about it.

## 30.03.20

#### Sentiment Analysis with Contextual Embeddings and Self-Attention
Authors: Katarzyna Biesialska, Magdalena Biesialska and Henryk Rybinski.  
Release Date: 12.03.20

The paper presents the Transformer-based Sentiment Analysis (TSA), a multilingual sentiment 
classifier model that is comparable or even outperforms state-of-the-art models for languages like 
Polish and German. The model is based on the Transformer architecture with a attention-based layer.

## 19.07.2020

#### Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data
Authors: Jacson Rodrigues, Rodrigo F. Berriel, ..., Alberto F. de Souza, Thiago Oliveira-Santos
Release Date: 14.06.2020

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
Release Date: 08.05.2020

The paper proposes a autoencoder network called High-Level Representation Guided Denoiser (HGD) as 
a defense for image classification. This network acts as a defender for a target network. It 
generalizes well to other unseen classes and transferability to other target networks.

The autoencoder proposed is a modified version of DAE (denoiser autoencoder) with U-net structure. 
It has lateral connections from encoder layers to their corresponding decoder layers and the 
learning objective is the adversarial noise.