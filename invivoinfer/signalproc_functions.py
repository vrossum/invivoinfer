from __future__ import division
import numpy as np
import scipy.signal as ssig

import math
import random
import matplotlib.pyplot as plt

import pdb

import nitime.algorithms as tsa
import nitime.utils as utils



# --------------------   Generate Inhomogeneous Poisson Process time events

def power_spectrum(trace,dt,windowl,overlap_perc=0.75,plot=0):
    
    # using welch method
    window_bin=nextpow2(windowl/dt)
    overlap_bin=int(window_bin*overlap_perc)

    f, Pxx_den = ssig.welch(trace,fs=1/dt,window='hanning',nperseg=window_bin,noverlap=overlap_bin, detrend='constant', scaling='density')

    Pxx_den_std=Pxx_den*(window_bin/trace.size *11 / 9)**0.5

    #pdb.set_trace()
    if plot==1:
        plt.semilogy(f, Pxx_den)
        plt.semilogy(f,Pxx_den+Pxx_den_std,color='red')
        plt.semilogy(f,Pxx_den-Pxx_den_std,color='red')
        plt.ylim([0.5e-5, 1])
        plt.xlim([0, 300])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()

    return f,Pxx_den,Pxx_den_std







# ------------ USED HERE
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n
