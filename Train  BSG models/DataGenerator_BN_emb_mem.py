import tensorflow
from tensorflow import keras
import numpy as np
import librosa
import os
import augmentation as aug

# DATA GENERATOR

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_to_data, data_paths, labels, weights, batch_size=32, dim=144000, n_classes=4, shuffle=True, augment=False, augm_params={}, p_mixup=0, p_dist_effect=0, TFLITE_THREADS = 1):
        self.data={}
        self.path = path_to_data
        self.data_paths = data_paths
        self.labels = labels
        self.weights = weights
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment 
        self.augm_params = augm_params
        self.p_mixup = p_mixup
        self.p_dist_effect = p_dist_effect
        self.MODEL_PATH: str = 'BirdNET-Analyzer-main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
        self.TFLITE_THREADS = TFLITE_THREADS # can be as high as number of CPUs
        # Initialize BirdNET feature extractor
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        PBMODEL = None
        C_PBMODEL = None
        self.INTERPRETER = tflite.Interpreter(model_path=self.MODEL_PATH, num_threads=self.TFLITE_THREADS)
        self.INTERPRETER.allocate_tensors()
        # Get input and output tensors.
        input_details = self.INTERPRETER.get_input_details()
        output_details = self.INTERPRETER.get_output_details()
        # Get input tensor index
        self.INPUT_LAYER_INDEX = input_details[0]["index"]
        # Get classification output or feature embeddings
        self.OUTPUT_LAYER_INDEX = output_details[0]["index"] - 1
        # load data to memory
        print("Loading data...")
        for i in range(len(self.data_paths)):
            sig, sr = librosa.load(self.path + self.data_paths[i], sr=48000, mono=True, res_type='kaiser_fast')
            if(len(sig) < self.dim): #pad too short clips
                sig = aug.pad(sig, 0, len(sig)/48000, target_len=self.dim, sr=48000)
            self.data[self.data_paths[i]] = sig
            if i % 100 == 0:
                print(f"Processing clip {i}/{len(self.data_paths)}...   ", end='\r')
        print("P")
        print("Data loading complete.")
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data_paths_temp = [self.data_paths[k] for k in indexes]
        X, y, w = self.__data_generation(data_paths_temp)
        return X, y, w

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def embeddings(self, sample):
        # Reshape input tensor
        self.INTERPRETER.resize_tensor_input(self.INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
        self.INTERPRETER.allocate_tensors()

        # Extract feature embeddings
        self.INTERPRETER.set_tensor(self.INPUT_LAYER_INDEX, np.array(sample, dtype="float32"))
        self.INTERPRETER.invoke()
        features = self.INTERPRETER.get_tensor(self.OUTPUT_LAYER_INDEX)
        return features

    def __data_generation(self, data_paths_temp):
        # Initialization
        X = np.empty((self.batch_size, self.dim), dtype='float32')
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        weights = np.empty(self.batch_size)        
        # Load and augment data samples 
        for i, ID in enumerate(data_paths_temp):
            label = self.labels[ID]
            weight = self.weights[ID]
            sig = self.data[ID]
            apply_mixup = 0
            mixup_signal = []
            if self.augment:
                if np.random.rand() < self.p_mixup: # Prepare file for mixup
                    apply_mixup = True
                    mixup_ID = np.random.choice(self.data_paths)
                    mixup_signal, sr = librosa.load(self.path + mixup_ID, sr=48000, mono=True, res_type='kaiser_fast')
                    if(len(mixup_signal) < self.dim): # pad too short clips
                        mixup_signal = aug.pad(mixup_signal, 0, len(mixup_signal)/48000, target_len=self.dim, sr=48000)
                    start = np.random.randint(0, np.max([1, len(mixup_signal)-self.dim]))
                    mixup_signal = mixup_signal[start:start+self.dim]
                    label = label + self.labels[mixup_ID]
                    label = [np.min((j, 1)) for j in label]
                    weight = (weight + self.weights[mixup_ID])/2
                # Augmentation
                sig = aug.augmentation(sig, p_dist_effect=self.p_dist_effect, mixup_signal=mixup_signal, apply_mixup=apply_mixup, **self.augm_params)
            # Store data and label
            if len(sig)>self.dim:
                start = np.random.randint(0, len(sig)-self.dim)
                sig = sig[start:start+self.dim]           
            X[i,:] = sig
            y[i,] = label
            weights[i] = weight
        X = self.embeddings(X)
        return X, y, weights
