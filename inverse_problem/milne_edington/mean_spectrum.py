import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from inverse_problem.milne_edington import compute_mean_spectrum

filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
batch_size = 160000
nbatches = 20


def compute_save(filename, batch_size, nbatches):
    spectrum_dict = compute_mean_spectrum(filename, batch_size=batch_size, nbatches=nbatches)
    project_path = Path(__file__).resolve().parents[1].parent
    filename = os.path.join(project_path, 'data/spectrumRanges.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(spectrum_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return spectrum_dict


if __name__ == '__main__':
    spectrum_dict = compute_save(filename, batch_size, nbatches)
    project_path = Path(__file__).resolve().parents[1].parent
    filename = os.path.join(project_path, 'data/spectrumRanges.pickle')
    with open(filename, 'rb') as handle:
        spectrum_dict = pickle.load(handle)
    plt.subplot(2, 2, 1)
    plt.plot(spectrum_dict['mean'][:56])
    plt.plot(spectrum_dict['max'][:56])
    plt.plot(spectrum_dict['min'][:56])

    plt.subplot(2, 2, 2)
    plt.plot(spectrum_dict['mean'][56:2 * 56])
    plt.plot(spectrum_dict['max'][56:2 * 56])
    plt.plot(spectrum_dict['min'][56:2 * 56])
    plt.subplot(2, 2, 3)
    plt.plot(spectrum_dict['mean'][2 * 56:3 * 56])
    plt.plot(spectrum_dict['min'][2 * 56:3 * 56])
    plt.plot(spectrum_dict['max'][2 * 56:3 * 56])
    plt.subplot(2, 2, 4)
    plt.plot(spectrum_dict['mean'][3 * 56:])
    plt.plot(spectrum_dict['min'][3 * 56:])
    plt.plot(spectrum_dict['max'][3 * 56:])
    plt.show()
