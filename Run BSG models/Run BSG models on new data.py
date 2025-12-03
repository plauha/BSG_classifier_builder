import pandas as pd
import numpy as np
import os
import gc
import tensorflow as tf
from tensorflow import keras
from classifier import Classifier
from functions import calibrate, threshold_filter

threshold = 0.3 # only save predictions with confidence higher than threshold
model_folder='models/Finland/model_v4/'
model_name='BSG_birds_Finland_v4_4.keras' 

# load classification model
# TFLITE_THREADS can be as high as number of CPUs available, the rest of the parameters should not be changed
clsf = Classifier(path_to_model= model_folder + model_name, sr=48000, clip_dur=3.0, TFLITE_THREADS = 1, offset=0, dur=0) 

# load species list and post-processing tables for prediction calibration
sp_list=pd.read_csv(model_folder + 'classes.csv')
# load calibration information (if it exists)
try:
    cal_table = np.load(model_folder + 'calibration_params.npy')
except:
    print(f"Could not load calibration data for model {model_name}")
    
# define path to audio data
path = "test_audio"

# analyze all files
files = os.listdir(path)

with open(path + '_results.txt', 'a') as f:
    f.write("site, file, species, prediction, detection_time \n")

n_files = len(files)
for j, fi in enumerate(files):
    try:
        print(f"Analyzing {fi} ({j+1}/{n_files})...")
        # predict for example clip
        pred, t = clsf.classify(path + '/' + fi, max_pred=False) #max_pred: only keep highest confidence detection for each species instead of saving all detections
        # calibrate prediction (only possible if calibration data exists)
        for i in range(len(pred)):
            pred[i, :] = calibrate(pred[i, :], cal_table=cal_table)
        # ignore human and noise predictions
        pred[:,0:2] = 0 
        # filter predictions with a threshold 
        pred, c, t = threshold_filter(pred, t, threshold)
        # filter and find species names from sp_list
        for i in range(len(pred)):
            if c[i] > 1: # ignore two first classes: noise and human
                with open(path + '_results.txt', 'a') as f:
                    f.write(path + ", " + fi + ", " + str(sp_list['common_name'].iloc[c[i]]) + ", " + str(pred[i]) + ", " + str(t[i]) + "\n")
        gc.collect() # clear memory
    except: 
        print(f"Error analyzing {fi}!")

print(" ")
print("All files analyzed")
print(f"Results saved to {path}_results.txt")