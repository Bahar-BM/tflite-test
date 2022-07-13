#!/usr/bin/env python3
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

######## TEST - crash pattern  #########################
x0 =  Input(shape=(23, 512))
x1 = tf.math.reduce_sum(x0, axis=-1)

model = Model([x0], [x1], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('reduce_sum.h5')
model.save('reduce_sum')
