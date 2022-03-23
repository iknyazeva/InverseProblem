import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys
import MEbatch_hs

wl0 = 6302.5
g = 2.5
mu = 1
l_v = [wl0, g, mu]
argument = np.linspace(6302.0692255, 6303.2544205, 56)
line_arg = 1000*(argument - wl0)


class point():
    def __init__(self, obs_prof, bbs, refer_p0, initial_mode = 0, l_v = [wl0, g, mu], line_arg = 1000*(argument - wl0)):
        self.l_v = l_v
        self.line_arg = np.reshape(line_arg, (1, -1))
        self.obs_prof = obs_prof
        self.initial_mode = initial_mode
        self.bounds = bbs[:2]
        self.sigmas = bbs[2]
        self.refer_p0 = refer_p0
        
    def ME(self, line_arg, *args):
        params = np.array([args])
        params = np.reshape(params, (1, -1))
        profile = MEbatch_hs.ME_ff(self.l_v, params, line_arg)
        return profile[0,:,:].T.flatten()
    
    def loss(self, p_v):
        resid = self.obs_prof - self.ME(*p_v)
        return resid.flatten()
    
    def find_p0(self):
        if self.initial_mode == 0:
            self.p0 = self.refer_p0
            return self.p0
        if self.initial_mode == 1:
            p0 = self.refer_p0 + self.sigmas*np.random.normal(11)
            p0 = np.max( np.array([p0, self.bounds[0]]), axis = 0)
            p0 = np.min( np.array([p0, self.bounds[1]]), axis = 0)
            self.p0 = self.refer_p0
            return self.refer_p0

    def find_opt(self):
        try:
            out = scipy.optimize.curve_fit(self.ME, xdata = self.line_arg, ydata = self.obs_prof.flatten(), p0 = self.find_p0())
            self.opt = out[0]
            self.opt = np.max([self.bounds[0], self.opt], axis = 0)
            self.opt = np.min([self.bounds[1], self.opt], axis = 0)
            self.cov = out[1]
        except:
            self.opt = self.find_p0()
            self.cov = np.zeros((11, 11))
        
    def check(self):
        plt.plot(self.obs_prof)
        plt.plot(self.light_ME(*self.find_opt()))
        plt.plot(self.light_ME(*self.find_p0()), linestyle = 'dashed')
        print(self.find_opt())
        print(self.find_p0())
        
