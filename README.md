# Adversarial Autoencoders

- Tensorflow implementation of [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) (ICLR 2016)

##  
### Usage
 Training. Summary, randomly sampled images and latent space will be saved in `SAVE_PATH`.

 ```
 python aae_mnist.py --train --dist_type <TYPE_OF_PRIOR>
 ```
 
 Random sample data from trained model. Image will be saved in `SAVE_PATH` with name `generate_im.png`.
 ```
 python aae_mnist.py --generate --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 Visualize latent space and data manifold (only when code dim = 2). Image will be saved in `SAVE_PATH` with name `generate_im.png` and `latent.png`. For Gaussian distribution, there will be one image for data manifold. For mixture of 10 2D Gaussian, there will be 10 images of data manifold for each component of the distribution.
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

### Architecture
*Prior Distribution* | *Description*
:---: | :--- |
<img src = 'figs/s_1.png' width = '1200px'> | The top row is an autoencoder. z is sampled through the reparameterization trick discussed in [variational autoencoder paper](https://arxiv.org/abs/1312.6114). The bottom row is a discriminator to separate samples generate from the encoder and samples from the prior distribution p(z).


### Hyperparameters
*name* | *value* |
:---| :---|
Dimention of z | 2 |
Batch Size | 128 |
Max Epoch | 400 |
Learning Rate | 2e-4 initial / 2e-5 after 100 epochs / 2e-6 after 300 epochs
Reconstruction Loss Weight | 1.0 |
Letant z Generator and Discriminator Loss Weight | 6.0 / 6.0 |

### Result
*Prior Distribution* | *Learned Coding Space* | *Learned Manifold*
:---: | :---: | :---: |
<img src = 'figs/gaussian.png' height = '230px'> | <img src = 'figs/gaussian_latent.png' height = '230px'> | <img src = 'figs/gaussian_manifold.png' height = '230px'>
<img src = 'figs/gmm.png' height = '230px'> | <img src = 'figs/gmm_latent.png' height = '230px'> | <img src = 'figs/gmm_manifold.png' height = '230px'>

### Incorporating label in the Adversarial Regularization

### Architecture
*Prior Distribution* | *Description*
:---: | :--- |
<img src = 'figs/s_2.png' width = '1200px'> | The only difference from previous model is that the one-hot label is used as input of encoder and there is one extra class for unlabeled data. For mixture of Gaussian prior, real samples are drawn from each components for each labeled class and for unlabeled data, real samples are drawn from the mixture distribution.

### Hyperparameters
Hyperparameters are the same as previous section.

### Usage
 Training. Summary, randomly sampled images and latent space will be saved in `SAVE_PATH`.

 ```
 python aae_mnist.py --train --label --dist_type <TYPE_OF_PRIOR>
 ```
 
 Random sample data from trained model. Image will be saved in `SAVE_PATH` with name `generate_im.png`.
 ```
 python aae_mnist.py --generate --label --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
  Visualize latent space and data manifold (only when code dim = 2). Image will be saved in `SAVE_PATH` with name `generate_im.png` and `latent.png`. For Gaussian distribution, there will be one image for data manifold. For mixture of 10 2D Gaussian, there will be 10 images of data manifold for each component of the distribution.
 ```
 python aae_mnist.py --viz --label --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 
*Number of Label Used* | *Learned Coding Space* | *Learned Manifold*
:--- | :---: | :---: |
**Use full label**| <img src = 'figs/gmm_full_label.png' width = '350px'> | <img src = 'figs/gmm_full_label_2.png' height = '150px'> <img src = 'figs/gmm_full_label_1.png' height = '150px'><img src = 'figs/gmm_full_label_0.png' height = '150px'> <img src = 'figs/gmm_full_label_9.png' height = '150px'>
**10k labeled data and 40k unlabeled data** | <img src = 'figs/gmm_10k_label.png' width = '350px'> | <img src = 'figs/gmm_10k_label_2.png' height = '150px'> <img src = 'figs/gmm_10k_label_1.png' height = '150px'><img src = 'figs/gmm_10k_label_0.png' height = '150px'> <img src = 'figs/gmm_10k_label_9.png' height = '150px'>

### Supervised Adversarial Autoencoders
*Code Dim=2* | *Code Dim=10* | *Code Dim=2*
:---: | :---: | :---: |
<img src = 'figs/supervise_code2.png' height = '230px'>| <img src = 'figs/supervise_code10.png' height = '230px'>| <img src = 'figs/supervise_code2.png' height = '230px'>

### Semi-supervised learning
*name* | *value* |
:---| :---|
Dimention of z | 10 |
Batch Size | 128 |
Max Epoch | 400 |
Learning Rate | 1e-4 initial / 1e-5 after 50 epochs / 1e-6 after 150 epochs
Reconstruction Loss Weight | 1.0 |
Letant z Generator and Discriminator Loss Weight | 6.0 / 6.0 |
Letant y Generator and Discriminator Loss Weight | 6.0 / 6.0 |

