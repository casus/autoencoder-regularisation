# autoencoder-regularisation-
Legendre-Latent-Space Regularisation ensures Toplogical Data-Structure Preservation under Autoencoder Compression

This repo contains code and supplemenatry material of the corresponding article, available here: TBA

## datasets considered
* Fashion MNIST
* MRI brain scans : [Open Access Series of Imaging Studies (OASIS)](https://oasis-brains.org/#data)
* synthetic datasets of points on highdimensional circle and torus 

## The repository consists of the following autoencoder models: 

* MLP-AE : Multilayer perceptron autoencoder
* AE-REG : Regularized autoencoder (proposed AE with Jacobian regularization)
* Hybrid AE-REG : Hybrid regularized autoencoder (Proposed AE with hybridization through orthogonal polynomial interpolation and Jacobian regularization)
* CNN-AE : Convolutional neural network autoencoder
* Contra AE : Contractive Autoencoder  
* MLP-VAE : Multilayer perceptron based variational autoencoder
* CNN-VAE : Convolutional neural network based variational autoencoder

## Orthogonal polynomial regresssion step for Hybrid AE-REG prior to training

This step prior to training of the proposed Hybrid AE-REG involves fitting involves extraction of the coefficients for the fitted orthogonal polynomial series

Before the running any files in the repository change the directory to root using cd ./autoencoder-regularisation-