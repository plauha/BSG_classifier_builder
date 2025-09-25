import numpy as np
import librosa 
import scipy
import colorednoise as cn #https://pypi.org/project/colorednoise/
import noisereduce as nr #https://pypi.org/project/noisereduce/ 

# Apply random augmentations on input signal

def augmentation(signal, # input signal
                 p_dist_effect = 0.8, # probability to apply distance_effect()
                 ir_matrix = [], # impulse response matrix for distance_effect()
                 ir_distances = [], # impulse response distances for distance_effect()
                 p_pitch_shift = 0.5, # probability to apply pitch_shift()
                 pitch_shift_steps = 1, # standard deviation of steps for pitch_shift()
                 p_time_stretch = 0.5, # probability to apply time_stretch()
                 time_stretch_rate = 0.08, # standard deviation of rate for time_stretch()
                 p_noise_red = 0.4, # probability to remove noise
                 noise_beta_params = [5,2], # beta params for reduce_noise()
                 apply_mixup = False, # apply mixup()
                 mixup_signal = [], # signal to mix with
                 mixup_norm_std = 15, # standard deviation of signal ratio for mixup()
                 mask_lambda = 0.5, # lambda parameter to define the number of masks
                 mask_max_length = 0.25, # maximum length of mask mask in proportion to the length of the clip
                 noise_gen_beta_params = [1,1], # beta parameters for noise color
                 p_resample = 0.5, #probability to apply resample()
                 resample_target_srs = [16000, 22050, 24000], # target sampling rates for resampling
                 target_len = 3*48000, # final length of the clip
                 orig_sr = 48000 # original sampling rate
                 ):

    signal_out = signal.copy()
    try:
        if(np.random.rand() < p_dist_effect):
            n_b = np.random.beta(noise_gen_beta_params[0], noise_gen_beta_params[1])+1
            signal_out = distance_effect(signal=signal_out, ir_matrix=ir_matrix, ir_dist=ir_distances, noise_beta=n_b)
    except:
        print(" Augmentation failed after dist_effect")
    try:
        if(np.random.rand() < p_pitch_shift):
            signal_out = pitch_shift(signal=signal_out, steps_sd=pitch_shift_steps, sr=orig_sr)
    except:
        print(" Augmentation failed after pitch_shift")
    try:
        if(np.random.rand() < p_time_stretch):
            n_b = np.random.beta(noise_gen_beta_params[0], noise_gen_beta_params[1])+1
            signal_out = time_stretch(signal_out, rate_sd=time_stretch_rate, target_length=target_len, noise_beta=n_b) 
        if len(signal_out)>target_len: # clip to target_length after time_stretch
            start = np.random.randint(0, len(signal_out)-target_len)
            signal_out = signal_out[start:start+target_len]
    except:
        print(" Augmentation failed after time_stretch")
    try:
        if(np.random.rand() < p_noise_red): 
            signal_out = reduce_noise(signal=signal_out, beta_params=noise_beta_params, sr=orig_sr)
    except:
        print(" Augmentation failed after noise_reduce")
    try:
        if(apply_mixup):
            signal_out = mixup(signal_out, signal2=mixup_signal, norm_std=mixup_norm_std, norm_mean=0)
            n_b = np.random.beta(noise_gen_beta_params[0], noise_gen_beta_params[1])+1
            signal_out = mask(signal_out, lamb=mask_lambda, max_length=mask_max_length, noise_beta=n_b)
    except:
        print(" Augmentation failed after mixup")
    try:
        if(np.random.rand() < p_resample):
            signal_out2 = resample(signal_out, target_srs=resample_target_srs, orig_sr=orig_sr)
            if np.isnan(signal_out2).any():
                a = 1
            else:
                signal_out = signal_out2
    except:
        a = 1
    return signal_out

##################################    
###### AUGMENTATION METHODS ######    
##################################

# add distance effect
def distance_effect(signal, ir_matrix, ir_dist, noise_beta=1):
    ind = np.random.randint(ir_matrix.shape[0]) # select random distance
    dist = ir_dist[ind]
    signal_out = scipy.signal.fftconvolve(signal, ir_matrix[ind], mode='same') # distance effect through convolution
    snr_start = np.random.normal(30, 10) # select snr from close distance (typically between 10-50)
    noise = cn.powerlaw_psd_gaussian(noise_beta, len(signal)) # generate random white/pink noise
    snr = snr_start-6*np.log2(dist) # signal attenuates 6 dB per doubling the distance
    signal_out = mix_signals(signal, noise, snr) # mix signals
    return signal_out

# change signal pitch
def pitch_shift(signal, steps_sd=1, sr=48000):
    # signal: input audio signal, steps_sd: standard deviation for the normal distribution defining the number of steps in pitch shift, sr: sampling rate
    steps = np.random.normal(0,steps_sd)
    signal_out = librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)
    return signal_out

# make the signal faster or slower
def time_stretch(signal, rate_sd=0.1, target_length=3*48000, noise_beta=1):
    # signal: input audio signal, rate_sd: standard deviation for the normal distribution defining the rate of fastening/slowening, target_length: the minimum length for the clip after stretching, noise_beta: the type of noise used for padding too short results (0=white noise, 1=pink noise, 2=red noise)
    length = len(signal)
    rate = np.random.normal(1,rate_sd)
    sig = librosa.effects.time_stretch(signal, rate=rate)
    if len(sig)<target_length: #pad with noise
        noise = cn.powerlaw_psd_gaussian(noise_beta, target_length)
        signal_out = noise*signal_scale(signal, noise, 30) # snr 30
        start = round((target_length-len(sig))/2)
        signal_out[start:start+len(sig)]=sig
        return signal_out
    else:
        return sig

# reduce background noise
def reduce_noise(signal, beta_params=[5,2], sr=48000):
    # signal: input audio signal, beta_params: parameters for beta distribution to define the proportion of noise reduce (0-100%)
    prop = np.random.beta(beta_params[0], beta_params[1])
    signal_out = nr.reduce_noise(y=signal, sr=sr, prop_decrease=prop)
    return signal_out

# mixup of two signals
def mixup(signal1, signal2, norm_std=10, norm_mean=0):
    # signal1, signal2: signals to be mixed, norm_std, norm_mean: parameters for the normal distribution to define the ratio between signals 
    ratio = np.random.normal(norm_mean, norm_std) # define the dB ratio between the signals
    signal_out = mix_signals(signal1, signal2, ratio)
    return signal_out

# mask random time intervals by replacing the signal with noise
def mask(signal, lamb=0.5, max_length=0.25, noise_beta=1):
    # signal: input audio signal, snr: signal-to-noise-ratio between signal and masks, lam: lambda-parameter for number of masks, max_length: maximum length of a single mask, noise_beta: the type of noise (0=white noise, 1=pink noise, 2=red noise)
    signal_out = signal.copy()
    # create background noise
    noise = cn.powerlaw_psd_gaussian(noise_beta, len(signal))
    snr = np.random.normal(20, 5) # snr typically between 10-30
    noise = noise*signal_scale(signal, noise, snr)
    # mask random parts of input signal
    l = len(signal)
    r = np.random.poisson(lamb) # sample random number of masks
    if(r > 0):
        for j in range(r):
            start,stop = np.sort((j*l/r)+np.random.randint(0, l/r, 2))
            if(np.random.rand()<0.5): # shorten too long masks
                stop = np.min([stop, start + max_length*l])
            else:
                start = np.max([start, stop - max_length*l])
            signal_out[int(start):int(stop)] = noise[int(start):int(stop)] # set the mask
    return signal_out

# downsample the signal and resample back to original scale
def resample(signal, target_srs=[16000, 22050, 24000], orig_sr=48000):
    # signal_main: signal to be processed, target_sr: list of possible sample rates for downsampling, orig_sr: original sample rate
    target_sr = np.random.choice(target_srs)
    signal_out = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
    signal_out = librosa.resample(signal_out, orig_sr=target_sr, target_sr=orig_sr)
    return signal_out

# pad too short signal with noise
def pad(signal, x1, x2, snr=25, noise_beta=1.3, target_len=3*48000, sr=48000):
    # signal: input audio signal, x1: starting point in seconds x2: ending point in seconds, snr: signal-to-noise-ratio between signal and padding mask, noise_beta: the type of noise (0=white noise, 1=pink noise, 2=red noise), # target_len: target length for signal, sr: sampling rate
    noise = cn.powerlaw_psd_gaussian(noise_beta, target_len) 
    sig_out = noise*signal_scale(signal, noise, snr)
    sig_out[int(x1*sr):int(x2*sr)] = signal[int(x1*sr):int(x2*sr)]
    return sig_out

# help functions for scaling two signals

# define multiplier c for signal_b so that 10*log(a^2/c*b^2) = ratio
def signal_scale(signal_a, signal_b, ratio):
    # signal_a: first signal, signal_b: second signal, ratio: snr between signal a and b
    denom=(np.mean(signal_b**2) * 10**(ratio/10)) 
    if denom == 0:
        denom = 0.00000000001
    c = np.mean(signal_a**2)/denom 
    return np.sqrt(c) 

# mix signals a and b using given snr
def mix_signals(signal_a, signal_b, ratio):
    # signal_a: first signal, signal_b: second signal, ratio: snr between signal a and b
    c = signal_scale(signal_a, signal_b, ratio)
    signal_out = signal_a + c*signal_b
    denom=np.std(signal_out)
    if denom == 0:
        denom = 0.0000000001
    signal_out = (signal_out-np.mean(signal_out))/denom # normalize
    return signal_out
