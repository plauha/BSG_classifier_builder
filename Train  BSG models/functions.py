import numpy as np
from scipy.signal import butter, lfilter

# Bandpass filter and time clipping
def butter_bandpass(cutoff, fs, order=5):
    if cutoff[0] > 200: # for stability 
        return butter(order, cutoff, fs=fs, btype='bandpass', analog=False)
    else:
        return butter(order, cutoff[1], fs=fs, btype='lowpass', analog=False)

def butter_bandpass_filter(data, cutoff, fs, order=5):
    cutoff[1] = np.min([cutoff[1], fs/2])
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    if np.isnan(y).any(): # for stability
        b, a = butter_bandpass(cutoff, fs, order=4)
        y = lfilter(b, a, data)
    return y