# @Author: Michel Andy <ragnar>
# @Date:   2020-01-24T14:11:09-05:00
# @Email:  Andymic12@gmail.com
# @Filename: stacked_ae.py
# @Last modified by:   ragnar
# @Last modified time: 2020-01-25T10:50:28-05:00

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
from utils.plt_funcs import *
import sys
from datetime import datetime

NOW = datetime.utcnow().strftime('%Y%m%d%H%M%S')
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = '{}/'.format(ROOT_LOGDIR)
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

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3")  # tied weights
weights4 = tf.transpose(weights1, name="weights4")  # tied weights

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 150
writer = tf.contrib.summary.create_file_writer("/tmp/stacked_ae")
with writer.as_default():
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            #tf.summary.scalar("TRAIN MSE", 0.5, step=loss_train)
            tf.contrib.summary.scalar("Train MSE", loss_train)
            writer.flush()
            saver.save(sess, MODEL_DIR +"/stacked_ae.ckpt")

def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 9):
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


show_reconstructed_digits(X, outputs, MODEL_DIR+"/stacked_ae.ckpt")
save_fig("reconstruction_plot")
plt.show()
