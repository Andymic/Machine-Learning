# Author: Andy M.
# Last Modified: 5/1/20

import tensorflow as tf
import config
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])


def house_model(y_new):
    with config.SESS.as_default():
        xs = [1, 2, 3, 4, 5, 6]
        ys = [1, 1.5, 2, 2.5, 3, 3.5]
        model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(xs, ys, epochs=1000)
        return model.predict(y_new)[0]


prediction = house_model([7.0])
print(prediction)
