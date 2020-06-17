# VAE-LIME
Seminar "Explainable and Fair Machine Learning" - Summer Term 2020 - University of Tübingen  

## ToDo's  
* ~~Train the variational autoencoder on the german credit card dataset (``train_german_vae.py``)~~
* Replace the lime sampling with the trained vae.
* Rerun the adversarial attack:
    * Has the the PCA data distribution improved? 
    * Can VAE-LIME explain the adversarial attack?
* Repeat for other datasets

### Idea's
* Optimize Network structure, maybe try conv2D instead of dense
* Use different normalization for categorical and numerical features
---
## Sources:  
lime: [Link](https://github.com/marcotcr/lime)  
fooling lime: [Link](https://github.com/dylan-slack/Fooling-LIME-SHAP)  
pytorch vae example: [Link](https://github.com/pytorch/examples/tree/master/vae)  
