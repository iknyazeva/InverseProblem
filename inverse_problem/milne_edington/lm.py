import scipy
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import time

import me

def lm_inversion(spectrum, initial = 'mean', line_arg = None, line_vec = None):
    lower_bounds = np.array([0, 0, 0, 5, 0.1, 0.01, 0.01, 0.01, -20, 0, -20])
    upper_bounds = ([5000, 180, 180, 100, 2, 100, 1, 1, 20, 1, 20])
    
    if initial == 'mean':
        x0 = [1000, 45, 45, 30, 1, 10, 0.5, 0.5, 0, 0.5, 0]
    elif initial == 'random':
        x0 = lower_bounds + np.random.random(size = 11)*(upper_bounds - lower_bounds)
    else:
        raise
        
    fun = lambda x: np.power(me.me_model(x, line_arg = line_arg, line_vec = line_vec, with_ff = True, 
                                with_noise = False, cont = 1) - spectrum, 2).flatten()    
    params = scipy.optimize.least_squares(fun, x0 = x0, method = 'lm')
    
    return params.x
    
    
