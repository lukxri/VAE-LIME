# VAE-LIME
Seminar "Explainable and Fair Machine Learning" - Summer Term 2020 - University of Tübingen  

## Setup

```
conda creante -n efml_sem python=3.7 -y  
conda activate efml_sem  
pip install -r requirements.txt  
```  

## ToDo's  
* Train the variational autoencoder on the german credit card dataset (``train_german_vae.py``)
* *Later: Optimize Network structure, try other datasets.*
* Replace the lime sampling with the trained vae.
* Rerun the adversarial attack:
    * Has the the PCA data distribution improved? 
    * Can VAE-LIME explain the adversarial attack?

---
## Sources:  
lime: [Link](https://github.com/marcotcr/lime)  
fooling lime: [Link](https://github.com/dylan-slack/Fooling-LIME-SHAP)  
pytorch vae example: [Link](https://github.com/pytorch/examples/tree/master/vae)  
