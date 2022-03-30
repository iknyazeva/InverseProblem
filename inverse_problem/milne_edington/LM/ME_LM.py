import scipy
import numpy as np
import matplotlib.pyplot as plt
import skimage
from tqdm import tqdm
from astropy.io import fits
import point
import os
import time
import shutil
from numba import jit

def make_refer(refer):
    p = np.empty((refer[1].data.shape[0], refer[1].data.shape[1], 11))
    for j, k in enumerate([1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]):
        p[:,:, j] = refer[k].data
    return p

def bounds_and_sigmas(refer):
    fl_refer = np.reshape(refer, (-1, 11))
    bbs = np.array([np.min(fl_refer, axis = 0), np.max(fl_refer, axis = 0), np.std(fl_refer, axis = 0)])
    return bbs

config = np.loadtxt('.\config.txt', dtype = 'str', delimiter = '=')

spectra_directory = config[0][1]
refer_file = fits.open(config[1][1])
initial_mode = int(config[2][1])
output_directory = config[3][1]

localtime = time.localtime()
name = str(localtime[0]) + '_' + str(localtime[1]) + '_' + str(localtime[2]) + '_' + str(localtime[3]) + str(localtime[4])
try:
    os.mkdir(output_directory + name)
except:
    pass
out_path = output_directory + name + '/'

files_list = os.listdir(spectra_directory)
X = len(files_list)
Y = 0

shutil.copy('.\config.txt', out_path)

refer = make_refer(refer_file)
bbs = bounds_and_sigmas(refer)

parameters = np.empty_like(refer)

for X_count in tqdm(range(X)):
    spectra_sheet = fits.open(spectra_directory + files_list[X_count])
    if Y == 0:
        Y = spectra_sheet[0].data.shape[1]
        parameters = np.empty((X, Y, 11))
    
    for Y_count in tqdm(range(Y)):
        spectra = spectra_sheet[0].data[:, Y_count, 56:]
        spectra[0] *= 2
        p = point.point(spectra, bbs, refer[Y_count, X_count], initial_mode)
        p.find_opt()
        parameters[Y_count][X_count] = p.opt
        print(parameters[Y_count][X_count])

    np.savetxt(out_path + 'backup.txt', parameters.flatten())
        
hdul = fits.HDUList([fits.PrimaryHDU(parameters)])
hdul.writeto(out_path + 'output.fits', overwrite = 1) 
        


