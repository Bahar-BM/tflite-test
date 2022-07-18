#!/usr/bin/env python3
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

######## TEST - tf.stack  #########################
x00 = Input(shape=(23))
x01 = Input(shape=(23))

x1 = tf.stack([x00, x01], axis=2)

model = Model([x00, x01], [x1], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('test_stack.h5')
model.save('test_stack')