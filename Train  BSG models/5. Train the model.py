import os
import numpy as np
import scipy.io
import librosa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras.optimizers import RMSprop
import pickle
from DataGenerator_BN_emb_mem import DataGenerator
from classification_head import create_model
import augmentation as aug
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Fit model

# start with global data

path_to_model = 'models/Argentina_Chaco/'
path_out = 'models/Argentina_Chaco/model_v1/'

with_val = True # use validation set?
tflite_threads = 5 # number of cpus
epochs = 10
round=1

model_path_out = path_out + 'model_v1_' + str(round) + '.keras'
model_path_in = path_out + 'model_v1_' + str(round-1) + '.keras'

# Load data
path_to_data = path_to_model + 'global_training_data/'

with open(path_to_data + 'metadata/train_set.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(path_to_data + 'metadata/val_set.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open(path_to_data + 'metadata/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
with open(path_to_data + 'metadata/weights.pkl', 'rb') as f:
    weights = pickle.load(f)
with open(path_to_data + 'metadata/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Model hyperparameters
dim = 3*48000 
batch_size = 64
n_classes = len(classes.keys())

# Load impulse response matrix for distance effect augmentation
ir_matrix = scipy.io.loadmat('irmatrix/irmatrix.mat')['irmatrix'] 
ir_dist = np.load("irmatrix/distances.npy")

# Set up augmentation strategy
augmentation_params = {'ir_matrix':ir_matrix, 'ir_distances':ir_dist, 
                       'p_pitch_shift':0.3, 'pitch_shift_steps':1, 
                       'p_time_stretch':0.3, 'time_stretch_rate':0.08, 
                       'p_noise_red':0.3, 'noise_beta_params':[5,2], 
                       'mixup_norm_std':15,
                       'mask_lambda':0.5, 'mask_max_length':0.25,
                       'noise_gen_beta_params':[1,1], 
                       'p_resample':0.6, 'resample_target_srs':[16000, 22050, 24000], 
                       'target_len':dim, 'orig_sr':48000}
p_mixup = 0.3

# Generators 
training_generator = DataGenerator(path_to_data=path_to_data, data_paths=train_data, 
                                   labels=labels, weights=weights, batch_size=batch_size, dim=dim, 
                                   n_classes=n_classes, shuffle=True, augment=True, augm_params=augmentation_params, 
                                   p_mixup=p_mixup, p_dist_effect=1, TFLITE_THREADS=tflite_threads)

if round==1:
    model = create_model(n_classes)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=1e-3), 
                  metrics=['binary_accuracy'])
else: 
    model = tf.keras.models.load_model(model_path_in)

if with_val:
    validation_generator = DataGenerator(path_to_data=path_to_data, data_paths=val_data,  
                                         labels=labels, weights=weights, batch_size=batch_size, dim=dim, 
                                         n_classes=n_classes, shuffle=True, augment=False, TFLITE_THREADS=tflite_threads)
    history = model.fit(x = training_generator, validation_data = validation_generator, epochs = epochs)  
else:
    history = model.fit(x = training_generator, epochs = epochs) 
    
if(os.path.isdir(path_out)):
    model.save(model_path_out)
else:
    os.mkdir(path_out)
    model.save(model_path_out)

print("Model trained and saved")

np.save(path_out + 'training_history_' + str(round) + '.npy', history)

# continue with local data

with_val = True # use validation set?
tflite_threads = 5 # number of cpus
epochs = 15
round=2

model_path_out = path_out + 'model_v1_' + str(round) + '.keras'
model_path_in = path_out + 'model_v1_' + str(round-1) + '.keras'

# Load data
path_to_data = path_to_model + 'local_training_data/'

with open(path_to_data + 'metadata/train_set.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(path_to_data + 'metadata/val_set.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open(path_to_data + 'metadata/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
with open(path_to_data + 'metadata/weights.pkl', 'rb') as f:
    weights = pickle.load(f)
with open(path_to_data + 'metadata/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Model hyperparameters
dim = 3*48000 
batch_size = 64
n_classes = len(classes.keys())

# Load impulse response matrix for distance effect augmentation
ir_matrix = scipy.io.loadmat('irmatrix/irmatrix.mat')['irmatrix'] 
ir_dist = np.load("irmatrix/distances.npy")

# Set up augmentation strategy
augmentation_params = {'ir_matrix':ir_matrix, 'ir_distances':ir_dist, 
                       'p_pitch_shift':0.3, 'pitch_shift_steps':1, 
                       'p_time_stretch':0.3, 'time_stretch_rate':0.08, 
                       'p_noise_red':0.3, 'noise_beta_params':[5,2], 
                       'mixup_norm_std':15,
                       'mask_lambda':0.5, 'mask_max_length':0.25,
                       'noise_gen_beta_params':[1,1], 
                       'p_resample':0.6, 'resample_target_srs':[16000, 22050, 24000], 
                       'target_len':dim, 'orig_sr':48000}
p_mixup = 0.3

# Generators 
training_generator = DataGenerator(path_to_data=path_to_data, data_paths=train_data, 
                                   labels=labels, weights=weights, batch_size=batch_size, dim=dim, 
                                   n_classes=n_classes, shuffle=True, augment=True, augm_params=augmentation_params, 
                                   p_mixup=p_mixup, p_dist_effect=1, TFLITE_THREADS=tflite_threads)

if round==1:
    model = create_model(n_classes)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=1e-3), 
                  metrics=['binary_accuracy'])
else: 
    model = tf.keras.models.load_model(model_path_in)

if with_val:
    validation_generator = DataGenerator(path_to_data=path_to_data, data_paths=val_data,  
                                         labels=labels, weights=weights, batch_size=batch_size, dim=dim, 
                                         n_classes=n_classes, shuffle=True, augment=False, TFLITE_THREADS=tflite_threads)
    history = model.fit(x = training_generator, validation_data = validation_generator, epochs = epochs)  
else:
    history = model.fit(x = training_generator, epochs = epochs) 
    
if(os.path.isdir(path_out)):
    model.save(model_path_out)
else:
    os.mkdir(path_out)
    model.save(model_path_out)

print("Model trained and saved")

np.save(path_out + 'training_history_' + str(round) + '.npy', history)

# Observe training results

model_name = 'Argentina_Chaco/'
rounds = 2 # in how many runs was the model trained

# visualize results of network training
def plot_results(history, val = True):
    acc = history['binary_accuracy']
    loss = history['loss']
    if val:
        val_acc = history['val_binary_accuracy']
        val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    if val:
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training (and validation) accuracy')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if val:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training (and validation) loss')
    plt.legend()
    plt.grid()
    plt.show()

joint_history = {'loss':[], 'binary_accuracy':[], 'val_loss':[], 'val_binary_accuracy':[]}

for i in range(rounds):
    with open(model_name + 'training_history_' + str(i+1) + '.pkl', 'rb') as f:
        history = pickle.load(f)
        for stat in ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy']:
            joint_history[stat] = joint_history[stat] + history[stat]

plot_results(joint_history)