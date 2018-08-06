# Adversarial Autoencoders

- Tensorflow implementation of [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) (ICLR 2016)

##  
### Usage
 Training

 ```
 python aae_mnist.py --train --dist_type <TYPE_OF_PRIOR>
 ```
 
 Random sample data from trained model
 ```
 python aae_mnist.py --generate --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 Visualize latent space and data manifold (only when code dim = 2) 
 ```
 python aae_mnist.py --viz --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 <!---
*name* | *command* 
:--- | :---
Training |``python aae_mnist.py --train --dist_type <TYPE_OF_PRIOR>``|
Random sample data |``python aae_mnist.py --generate --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>``|
Visualize latent space and data manifold (only when code dim = 2) |``python aae_mnist.py --viz --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>``|
Option | ``--bsize``
--->

### Result
*Prior Distribution* | *Learned Coding Space* | *Learned Manifold*
:---: | :---: | :---: |
<img src = 'figs/gaussian.png' height = '230px'> | <img src = 'figs/gaussian_latent.png' height = '230px'> | <img src = 'figs/gaussian_manifold.png' height = '230px'>
<img src = 'figs/gmm.png' height = '230px'> | <img src = 'figs/gmm_latent.png' height = '230px'> | <img src = 'figs/gmm_manifold.png' height = '230px'>

### Incorporating label in the Adversarial Regularization
*Number of Label Used* | *Learned Coding Space* | *Learned Manifold*
:--- | :---: | :---: |
**Use full label**| <img src = 'figs/gmm_full_label.png' width = '350px'> | <img src = 'figs/gmm_full_label_2.png' height = '150px'> <img src = 'figs/gmm_full_label_1.png' height = '150px'><img src = 'figs/gmm_full_label_0.png' height = '150px'> <img src = 'figs/gmm_full_label_9.png' height = '150px'>
**10k labeled data and 40k unlabeled data** | <img src = 'figs/gmm_10k_label.png' width = '350px'> | <img src = 'figs/gmm_10k_label_2.png' height = '150px'> <img src = 'figs/gmm_10k_label_1.png' height = '150px'><img src = 'figs/gmm_10k_label_0.png' height = '150px'> <img src = 'figs/gmm_10k_label_9.png' height = '150px'>

### Supervised Adversarial Autoencoders
*Code Dim=2* | *Code Dim=10* | *Code Dim=2*
:---: | :---: | :---: |
<img src = 'figs/supervise_code2.png' height = '230px'>| <img src = 'figs/supervise_code10.png' height = '230px'>| <img src = 'figs/supervise_code2.png' height = '230px'>
