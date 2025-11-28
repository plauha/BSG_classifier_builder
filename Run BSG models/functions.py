import numpy as np

# filter and sort predictions based on threshold
def top_preds(prediction, timestamps, threshold=0.5):
    # prediction: classification model output (max results), timestamp: timestamps from model, threshold: threshold for filtering (0-1)
    cls= [idx for idx, val in enumerate(prediction) if val > threshold]
    prediction = np.array(prediction)[cls]
    ts = np.array(timestamps)[cls]
    if len(cls)>0:
        prediction, cls, ts = map(list, zip(*sorted(zip(prediction, cls, ts), reverse=True)))
    else:
        prediction=[]
        cls = []
        ts = []
    return prediction, cls, ts

# filter predictions based on threshold
def threshold_filter(preds, timestamps, threshold=0.5):
    # prediction: classification model output (all results), timestamp: timestamps from model, threshold: threshold for filtering (0-1)
    arg_where = np.where(preds>threshold)
    prediction = preds[arg_where]
    cls = arg_where[1]
    ts = timestamps[arg_where[0]]
    return prediction, cls, ts

# pad too short signal with zeros
def pad(signal, x1, x2, target_len=3*48000, sr=48000):
    # signal: input audio signal, x1: starting point in seconds x2: ending point in seconds, 
    # target_len: target length for signal, sr: sampling rate
    sig_out = np.zeros(target_len) 
    sig_out[int(x1*sr):int(x2*sr)] = signal[int(x1*sr):int(x2*sr)]
    return sig_out

# split input signal to overlapping chunks
def split_signal(sig, rate, seconds, overlap):
    # sig: input_signal, rate: sampling rate, seconds: target length in seconds,
    # overlap: overlap of consecutive frames in seconds, minlen: m
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]
        if len(split) < int(seconds * rate): # pad if clip is too short
            split = pad(split, 0, len(split)/rate, target_len=int(seconds*rate), sr=rate)     
        sig_splits.append(split)
    return sig_splits

# calibrate prediction
def calibrate(p, cal_table):
    return [1/(1+np.exp(-(cal_table[i, 0]+cal_table[i, 1]*pr))) for i, pr in enumerate(p)]


    