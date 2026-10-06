# convert keras model to tflite for easier distribution

import tensorflow as tf

model_path = 'models/Argentina_Chaco/BSG_birds_Argentina_v1_2.keras'
keras_model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model) 
tflite_model = converter.convert()
with open(model_path.replace('keras', 'tflite'), 'wb') as f:     
  f.write(tflite_model)