# Author: Andy M.
# Last Modified: 9/11/20
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import numpy as np
import wget

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

if not os.path.exists('/tmp/rps'):
    wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip', '/tmp/rps.zip')
    wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip',
                  '/tmp/rps-test-set.zip')

rps_dirs = ['/tmp/rps.zip', '/tmp/rps-test-set.zip']
for rps_dir in rps_dirs:
    zip_ref = zipfile.ZipFile(rps_dir, 'r')
    zip_ref.extractall('/tmp/')
    zip_ref.close()

rock_dir = '/tmp/rps/rock'
paper_dir = '/tmp/rps/paper'
scissors_dir = '/tmp/rps/scissors'

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

pic_index = 2
next_rock = [os.path.join(rock_dir, fname)
             for fname in rock_files[pic_index - 2:pic_index]]
next_paper = [os.path.join(paper_dir, fname)
              for fname in paper_files[pic_index - 2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname)
                 for fname in scissors_files[pic_index - 2:pic_index]]

# for i, img_path in enumerate(next_rock+next_paper+next_scissors):
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
#   plt.axis('Off')
#   plt.show()

TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=20
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=20
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
model_path = '/tmp/rps.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
else:
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data=validation_generator, verbose=1,
                        validation_steps=3)

    model.save(model_path)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()

if not os.path.exists('/tmp/rps-validation'):
    wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip',
                  '/tmp/rps-validation.zip')

    zip_ref = zipfile.ZipFile('/tmp/rps-validation.zip', 'r')
    zip_ref.extractall('/tmp/rps-validation')
    zip_ref.close()

# file_path = raw_input('Enter file path:')

file_paths = os.listdir('/tmp/rps-validation')
random.shuffle(file_paths)

for i in range(0, 3):
    file_name = file_paths[i]
    img = image.load_img(os.path.join('/tmp/rps-validation',file_name), target_size=(150, 150))
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    print(file_name)
    classes = model.predict(images, batch_size=10)
    # classes[paper, rock, scissors]
    print(classes)