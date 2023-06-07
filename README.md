# prediction-gap
Fast and exact prediction gap calculation on tree models

## Setup
Create new environment using
```conda create --name <env_name> --file requirements.txt```

## About
Prediction gap on important feature perturbation (PGI) and predicition gap on unimportant feature perturbation (PGU) are two quantities that measure expected difference in model prediction when certain features of the input are modified with a random noise. They were first introduced in Agarwal et al. [1]. See therein for the definitions. The prediction gaps definitions don't use any knowledge about how the model operates (i.e., they work for a black box model). We show that when the model is based on decision trees, we can exploit its structure to calculate the prediction gaps faster and with reduced approximation error.

## Bibliography
1. Agarwal et al., *OpenXAI: Towards a Transparent Evaluation of Model Explanations*, 36th Conference on Neural Information Processing Systems (NeurIPS 2022), https://doi.org/10.48550/arXiv.2206.11104.
