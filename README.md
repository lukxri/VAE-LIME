# VAE-LIME
Seminar "Explainable and Fair Machine Learning" - Summer Term 2020 - University of Tübingen  

## ToDo's  
* Train the variational autoencoder on the german credit card dataset (``train_german_vae.py``)
* *Later: Optimize Network structure, try other datasets.*
* Replace the lime sampling with the trained vae.
* Rerun the adversarial attack:
    * Has the the PCA data distribution improved? 
    * Can VAE-LIME explain the adversarial attack?

Sources:  
pytorch vae example: [Link](https://github.com/pytorch/examples/tree/master/vae)
fooling lime : [Link](https://github.com/dylan-slack/Fooling-LIME-SHAP)
