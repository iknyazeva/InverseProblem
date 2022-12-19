# Code for "Stokes inversion techniques with neural networks: analysis of uncertainty in parameter estimation" paper

## Introduction

Over the past two decades, neural networks have been shown to be a fast and accurate alternative to classic inversion technique methods. However, most of these codes can be used to obtain point estimates of the parameters, so ambiguities, the degeneracies, and the uncertainties of each parameter remain uncovered.
We provide inversion codes based on the simple Milne-Eddington model of the stellar atmosphere and deep neural networks to both parameter estimation and their uncertainty intervals. 

The proposed framework is designed in such a way that it can be expanded and adapted to other atmospheric models or combinations of them. Additional information can also be incorporated directly into the model. 

## Data source
In the current study, we used a collection of the Level 1 calibrated Stokes spectra (comprised by images stored in FITS format) and collection of the Level 2 data sets (obtained from the MERLIN spectral line inversion of the Level 1 calibrated spectra) produced by the Spectropolarimeter (SP) on board the Hinode, since its launch in 2006. Hinode is a Japanese mission, developed and launched by ISAS/JAXA, with NAOJ as a domestic partner and NASA and STFC (UK) as international partners. It is operated by these agencies in co-operation with ESA and NSC (Norway). The Hinode has an open data policy, allowing anyone access to the data and data
products. Level 1 and 2 data are available by following the data link: https://csac.hao.ucar.edu/sp_data.php.


## Baseline model 

Preprint with the description: https://www.researchgate.net/project/Stockes-Spectral-Profile-Inversion-with-neural-networks.


`notebooks/baseline_mlp_partly_indep.ipynb`

## Run experiments
`notebooks/mlp_partly_indep_conv_unc.ipynb`

## Get figures

Predictions made on simulated data:
`notebooks/figures_simulated_data.ipynb`

Predictions made on real observations:
`notebooks/figures_real_spectra.ipynb`

