# Adversarial Autoencoders (AAE)

- Tensorflow implementation of [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) (ICLR 2016)

## Requirements
- Python 3.3+
- [TensorFlow 1.9+](https://www.tensorflow.org/)
- [TensorFlow Probability](https://github.com/tensorflow/probability)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)


## Implementation details
- All the models of AAE are defined in [src/models/aae.py](src/models/aae.py). **create**
- Examples of how to use AAE models can be found in [experiment/aae_mnist.py](experiment/aae_mnist.py).
- Encoder, decoder and all discriminators contain two fully connected layers with 1000 hidden units and RelU activation function. Decoder and all discriminators contain an additional fully connected layer for output.
- Images are normlized to [-1, 1] before fed into the encoder and tanh is used as the output nonlineary of decoder.

## Preparation
- Download the MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/).
- Setup path in [`experiment/aae_mnist.py`](experiment/aae_mnist.pyy):
`DATA_PATH ` is the path to put MNIST dataset.
`SAVE_PATH ` is the path to save output images and trained model.

## Usage
The script [experiment/aae_mnist.py](experiment/aae_mnist.py) contains all the experiments shown here. Detailed usage for each experiment will be describe later along with the results.
### Argument
* `--train`: Train AAE by imposing a prior distribution on the hidden codes without or with label (if `--label` is true).
* `--label`: AAE imposing a prior distribution on the hidden codes with label information.
* `--train_supervised`: Train the network.
* `--train_semisupervised`: Train the network.
* `--noise`:
* `--generate`: Random sample data from trained model.
* `--viz`: Visualize latent space and data manifold (only when `--ncode` is 2)
* `--load`:
* `--dist_type`: Type of the prior distribution use to impose on the hidden codes. Default: `gaussian`. `gmm` for Gaussian mixture distribution.
* `--ncode`: Code dimension. Default: `2`
* `--encw`:
* `--genw`:
* `--disw`:
* `--clsw`:
* `--ygenw`:
* `--ydisw`:
* `--lr`:
* `--dropout`:

## Adversarial Autoencoder

### Architecture
*Architecture* | *Description*
:---: | :--- |
<img src = 'figs/s_1.png' width = '1500px'> | The top row is an autoencoder. z is sampled through the reparameterization trick discussed in [variational autoencoder paper](https://arxiv.org/abs/1312.6114). The bottom row is a discriminator to separate samples generate from the encoder and samples from the prior distribution p(z).


### Usage

- Training. Summary, randomly sampled images and latent space during training will be saved in `SAVE_PATH`.

 ```
 python aae_mnist.py --train --ncode <CODE_DIM> --dist_type <TYPE_OF_PRIOR>
 ```
 
 - Random sample data from trained model. Image will be saved in `SAVE_PATH` with name `generate_im.png`.
 ```
 python aae_mnist.py --generate --ncode <CODE_DIM> --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 - Visualize latent space and data manifold (only when code dim = 2). Image will be saved in `SAVE_PATH` with name `generate_im.png` and `latent.png`. For Gaussian distribution, there will be one image for data manifold. For mixture of 10 2D Gaussian, there will be 10 images of data manifold for each component of the distribution.
 ```
 python aae_mnist.py --viz --ncode <CODE_DIM> --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 <!---
*name* | *command* 
:--- | :---
Training |``python aae_mnist.py --train --dist_type <TYPE_OF_PRIOR>``|
Random sample data |``python aae_mnist.py --generate --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>``|
Visualize latent space and data manifold (only when code dim = 2) |``python aae_mnist.py --viz --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>``|
Option | ``--bsize``
--->


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
- For 2D Gaussian, we can see sharp transitions (no gaps) as mentioned in the paper. Also, from the learned manifold, we can see almost all the sampled images are readable.
- For mixture of 10 Gaussian, I just uniformly sample images in a 2D square space as I did for 2D Gaussian instead of sampling along the axes of the corresponding mixture component, which will be shown in the next section. We can see in the gap area between two component, it is less likely to generate good samples.  

*Prior Distribution* | *Learned Coding Space* | *Learned Manifold*
:---: | :---: | :---: |
<img src = 'figs/gaussian.png' height = '230px'> | <img src = 'figs/gaussian_latent.png' height = '230px'> | <img src = 'figs/gaussian_manifold.png' height = '230px'>
<img src = 'figs/gmm.png' height = '230px'> | <img src = 'figs/gmm_latent.png' height = '230px'> | <img src = 'figs/gmm_manifold.png' height = '230px'>

## Incorporating label in the Adversarial Regularization

### Architecture
*Architecture* | *Description*
:---: | :--- |
<img src = 'figs/s_2.png' width = '1500px'> | The only difference from previous model is that the one-hot label is used as input of encoder and there is one extra class for unlabeled data. For mixture of Gaussian prior, real samples are drawn from each components for each labeled class and for unlabeled data, real samples are drawn from the mixture distribution.

### Hyperparameters
Hyperparameters are the same as previous section.

### Usage
- Training. Summary, randomly sampled images and latent space will be saved in `SAVE_PATH`.

 ```
 python aae_mnist.py --train --ncode <CODE_DIM> --label --dist_type <TYPE_OF_PRIOR>
 ```
 
- Random sample data from trained model. Image will be saved in `SAVE_PATH` with name `generate_im.png`.
 ```
 python aae_mnist.py --generate --ncode <CODE_DIM> --label --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 
- Visualize latent space and data manifold (only when code dim = 2). Image will be saved in `SAVE_PATH` with name `generate_im.png` and `latent.png`. For Gaussian distribution, there will be one image for data manifold. For mixture of 10 2D Gaussian, there will be 10 images of data manifold for each component of the distribution.
 ```
 python aae_mnist.py --viz --ncode <CODE_DIM> --label --dist_type <TYPE_OF_PRIOR> --load <RESTORE_MODEL_ID>
 ```
 ### Result
 - Compare with the result in the previous section, incorporating labeling information provides better fitted distribution for codes.
 - The learned manifold images demostrate that each Gaussian component corresponds to the one class of digit. However, the style representation is not consistently represented within each mixture component as shown in the paper. For example, the right most column of the first row experiment, the lower right of digit 1 tilt to left while the lower right of digit 9 tilt to right.

*Number of Label Used* | *Learned Coding Space* | *Learned Manifold*
:--- | :---: | :---: |
**Use full label**| <img src = 'figs/gmm_full_label.png' width = '350px'> | <img src = 'figs/gmm_full_label_2.png' height = '150px'> <img src = 'figs/gmm_full_label_1.png' height = '150px'><img src = 'figs/gmm_full_label_0.png' height = '150px'> <img src = 'figs/gmm_full_label_9.png' height = '150px'>
**10k labeled data and 40k unlabeled data** | <img src = 'figs/gmm_10k_label.png' width = '350px'> | <img src = 'figs/gmm_10k_label_2.png' height = '150px'> <img src = 'figs/gmm_10k_label_1.png' height = '150px'><img src = 'figs/gmm_10k_label_0.png' height = '150px'> <img src = 'figs/gmm_10k_label_9.png' height = '150px'>

### Supervised Adversarial Autoencoders

### Architecture
*Architecture* | *Description*
:---: | :--- |
<img src = 'figs/s_3.png' width = '1000px'> | The decoder takes code as well as a one-hot vector encoding the label as input. Then it forces the network learn the code independent of the label.

### Hyperparameters

### Usage
- Training. Summary and randomly sampled images will be saved in `SAVE_PATH`.

 ```
 python aae_mnist.py --ncode <CODE_DIM> --train_supervised
 ```
 
 - Random sample data from trained model. Image will be saved in `SAVE_PATH` with name `sample_style.png`.
 ```
 python aae_mnist.py --ncode <CODE_DIM> --generate --supervise --load <RESTORE_MODEL_ID>
 ```

### Result
- The result images are generated by using the same code for each column and the same digit label for each row.
- When code dimension is 2, we can see each column consists the same style clearly. But for dimension 10, we can hardly read some digits. Maybe there are some issues of implementation or the hyper-parameters are not properly picked, which makes the code still depend on the label. 

*Code Dim=2* | *Code Dim=10* | 
:---: | :---: | 
<img src = 'figs/supervise_code2.png' height = '230px'>| <img src = 'figs/supervise_code10.png' height = '230px'>| 

### Semi-supervised learning

### Architecture
*Architecture* | *Description*
:---: | :--- |
<img src = 'figs/s_4.png' width = '1000px'> | 

### Hyperparameters
*name* | *value* |
:---| :---|
Dimention of z | 10 |
Batch Size | 128 |
Max Epoch | 400 |
Learning Rate | 1e-4 initial / 1e-5 after 50 epochs / 1e-6 after 150 epochs
Reconstruction Loss Weight | 1.0 |
Letant z Generator and Discriminator Loss Weight | 6.0 / 6.0 |
Letant y Generator and Discriminator Loss Weight | 6.0 / 6.0 |


