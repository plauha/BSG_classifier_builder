from tensorflow import keras
from keras import layers
from keras.models import Model

# Create the classification head
def create_model(n_classes, shape=1024):
    inputs = layers.Input(shape=(shape,))
    hidden1 = layers.Dense(units=shape, activation='relu')(inputs)
    outputs = layers.Dense(units=n_classes, activation='sigmoid')(hidden1) 
    model = Model(inputs, outputs)
    return model
    
