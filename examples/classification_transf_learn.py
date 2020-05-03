# Author: Andy M.
# Last Modified: 5/2/20

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config
import tensorflow_datasets as tfds

# tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# print(raw_validation)
# print(raw_test)

get_label_name = metadata.features['label'].int2str

print('Label names: ', get_label_name)


def plot(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()


for image, label in raw_train.take(2):
    plot(image, label)

IMG_SIZE = 160  # All images will be resized to 160x160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
model = None
checkpoint_path = "training/ctl/classification_transf_learn.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_name = checkpoint_path + '.index'

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass

print('image_batch shape: ', image_batch.shape)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)
# print(feature_batch.shape)

# Freeze the convolutional base
base_model.trainable = False

# print(base_model.summary())
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])


def train():
    with config.SESS.as_default():
        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        print(model.summary())

        initial_epochs = 10
        validation_steps = 20

        loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        history = model.fit(train_batches,
                            epochs=initial_epochs,
                            validation_data=validation_batches,
                            callbacks=[cp_callback])

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()


def load(filenames):
    from PIL import Image
    from skimage import transform
    imgs = []
    for filename in filenames:
        img = Image.open(filename)
        img = (np.array(img).astype('float32') / 127.5) - 1
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


if os.path.exists(checkpoint_name):
    model.load_weights(checkpoint_path)
    test_imgs = ['dog_test.jpg', 'cat_test.jpg', 'dog_test1.jpg']
    test_imgs = load(test_imgs)
    print('test_imgs shape: ', test_imgs.shape)
    predictions = model.predict(test_imgs)

    for pred in predictions:
        pred = 0 if pred <= 0 else 1
        print(metadata.features['label'].names[pred])

else:
    train()
