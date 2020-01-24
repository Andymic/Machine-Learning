# @Author: Michel Andy <ragnar>
# @Date:   2020-01-24T14:11:09-05:00
# @Email:  Andymic12@gmail.com
# @Filename: stacked_ae.py
# @Last modified by:   ragnar
# @Last modified time: 2020-01-24T15:25:53-05:00

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
from util import *

MODEL_DIR='./models'

import os
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

mnist = input_data.read_data_sets("/tmp/data/")

n_inputs  = 28 * 28 #MNIST
n_hidden1 = 300
n_hidden2 = 150 #codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.001
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
dense_layer = partial(tf.layers.dense, #expects 2D
                    activation=tf.nn.elu,
                    kernel_initializer=he_init,
                    kernel_regularizer=l2_regularizer)

hidden1 = dense_layer(X, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)
outputs = dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) #MSE

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

loss = tf.add_n([reconstruction_loss] + reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples
        for iter in range(n_batches):
            print("\r{}%".format(100 * iter // n_batches), end="")
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, MODEL_DIR +"/stacked_ae.ckpt")

def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

show_reconstructed_digits(X, outputs, MODEL_DIR+"/my_model_all_layers.ckpt")
save_fig("reconstruction_plot")
