# Table of Contents:

## Data Files
### train_only.npy
Data file with the training instances only, in 120D format after extracting features with DWT.

### valid.npy
Data file with the training and validation instances, in 120D format after extracting features with DWT. 

## Demo Notebooks
### PGM_IGMM_V_3_0.ipynb
Demo code for running our implementation of IGMMs with different values for the priors.

## Implementation Files:
### igmm.py
Class which implements the logic for the collapsed gibbs sampling. 
### ui.py
Misc class for collecting the values of the priors from the user interface. Also includes a function for visualization, with plotly plus  functions implementing the dimensionality reduction from 120D to 2D using t-SNE.
### gaussian_components_diag.py
Class which implements the logic for computing the posterior predictive probability (p(X|z, mu, sigma etc.))