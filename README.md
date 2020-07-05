# VAE-LIME
Seminar "Explainable and Fair Machine Learning" - Summer Term 2020 - University of Tübingen  

## Setup
```
conda creante -n vae_lime python=3.7 -y  
conda activate vae_lime  
pip install -r requirements.txt  
```  

## Method

All of our contributions can be found in ``experiments/``.

* Train the variational autoencoder on the german credit card dataset (``train_german_vae.py``)
* Replace the lime sampling with the trained vae.
* Rerun the adversarial attack:
    * Has the the PCA data distribution improved? 
    * Can VAE-LIME explain the adversarial attack?
* Repeat for other datasets

The feature importance ranking plots where created [here](https://docs.google.com/spreadsheets/d/1xvP-7lWBjuXq95PpS93IuDAch26qeGMF8VSChvfy-MM/edit?usp=sharing)

### Ideas for future work
* Optimize Network structure, maybe try conv2D instead of dense
* Use different normalization and/or autoencoders for categorical and numerical features
---
## Sources:  
LIME github repository: [Link](https://github.com/marcotcr/lime)  
Fooling LIME github repository: [Link](https://github.com/dylan-slack/Fooling-LIME-SHAP)  
PyTorch Variational autoencoder example: [Link](https://github.com/pytorch/examples/tree/master/vae)  
