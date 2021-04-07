import scipy
from scipy.optimize import least_squares
import scipy.special
import numpy as np
from inverse_problem.milne_edington.me import me_model


def lm_inversion(spectrum, initial='mean', line_arg=None, line_vec=None):
    if not line_vec:
        line_vec = (6302.5, 2.5, 1)
    if not line_arg:
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])

    lower_bounds = np.array([0, 0, 0, 5, 0.1, 0.01, 0.01, 0.01, -20, 0, -20])
    upper_bounds = ([5000, 180, 180, 100, 2, 100, 1, 1, 20, 1, 20])

    if initial == 'mean':
        x0 = [1000, 45, 45, 30, 1, 10, 0.5, 0.5, 0, 0.5, 0]
    elif initial == 'random':
        x0 = lower_bounds + np.random.random(size=11) * (upper_bounds - lower_bounds)
    else:
        raise

    fun = lambda x: np.power(me_model(x, line_arg=line_arg, line_vec=line_vec, with_ff=True,
                                      with_noise=False, cont=1) - spectrum, 2).flatten()
    params = scipy.optimize.least_squares(fun, x0=x0, method='lm')

    return params.x


def predict_full_image(line, parameter, **kwargs):
    """ Predicts full image
    Args:
        line: array of size (n, 512)
        parameter (int): index of parameter to predict
    """
    output = np.zeros(line[0].shape[:2])
    for i in range(output.shape[0]):
        for t in range(output.shape[1]):
            predicted = lm_inversion(line[0][i, t].reshape((1, 56, 4)), **kwargs)
            output[i, t] = predicted[parameter]
    return output
