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

Before the running any files in the repository change the directory to root using `cd ./autoencoder-regularisation-`

* Run `python ./coefficients_computation_for_fitted_polynomials/FashionMNIST/parallel_0_to_10_dq25.py` to perform polynomial regression over Fashion MNIST dataset. Set `no_images` and `deg_quad` as required of keep the default values.
* Run `python ./coefficients_computation_for_fitted_polynomials/FashionMNIST/LSTSQparallel_fmnsit_train_dq20.py` to extract coefficients in parallel using multiple cores.

* Similarly run `python ./coefficients_computation_for_fitted_polynomials/MRI_scans/parallel_0_to_10.py` and other files in  `./coefficients_computation_for_fitted_polynomials/MRI_scans/` to extract fitted polynomial coefficients. 