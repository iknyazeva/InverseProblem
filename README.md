# Neural network with pytorch for Stockes Spectral Profile Inversion

Source code for solar atmospheric parameters recovering. See usage examples in notebook folder. 


## Why?

Spectropolarimetric observations are broadly used for the extraction of physical information in the field of solar physics. Inferring magnetic and thermodynamic information from these observations includes inversion problem solving. Assuming that spectropolarimetric profiles are produced by a given atmospheric model, it is required to find the best sets of parameters within such a model corresponding to particular observations. Standard optimization approach often requires large computational resources and even in this case still performs very slowly.

## What's new?
Previously it was suggested to use different strategies with artificial neural networks to overcome problems with computational power. It was previously shown that neural networks could be a viable alternative to the standard least square approach, but they could not replace it.  
Most papers only cover Magnetic Fields Vector parameter inferring, whereas the commonly used solar atmosphere model includes 11 parameters. In this paper we provide an end-to-end deep learning framework for full parameter inferring as well as comparison of several approaches for multi-output predictions. For this purpose, we trained one common network to predict all parameters, a set of parameter-oriented independent networks to deal with each parameter, and finally a combination of the above: a set of parameter-oriented independent networks built upon several layers of the pretrained common network. Our results show that using a partly independent network built upon a pretrained network provides the best results and demonstrates better generalization performance. 
