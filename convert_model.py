#!/usr/bin/python3

import tensorflow as tf

######## Conversion #########
tf_model = tf.keras.models.load_model('/model_files/test_stack.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model = converter.convert()
tflite_model_file = 'fp32_stack.tflite'

with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)


