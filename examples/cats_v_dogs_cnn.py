# Author: Andy M.
# Last Modified: 5/2/20


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import config
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

BATCH_SIZE = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_CHANNEL = 3

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImage(image):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plotGraph(train_cats_dir, train_dogs_dir):
    nrows = ncols = 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    
    pic_index+=8
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[ pic_index-8:pic_index]
                   ]
    
    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[ pic_index-8:pic_index]
                   ]
    
    for i, img_path in enumerate(next_cat_pix+next_dog_pix):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)
    
      img = mpimg.imread(img_path)
      plt.imshow(img)
    
    plt.show()


checkpoint_path = "training/cats_v_dogs/cats_v_dogs.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_name = checkpoint_path + '.index'

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

#plotGraph(train_cats_dir, train_dogs_dir)

image_gen_train = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=45,
                                     width_shift_range=.15,
                                     height_shift_range=.15,
                                     horizontal_flip=True,
                                     zoom_range=0.5)

image_gen_val = ImageDataGenerator(rescale=1. / 255)

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')

label_map = train_data_gen.class_indices
labels = dict((i,c) for c,i in label_map.items())

print('Classes: ', label_map)

# sample_training_images, _ = next(train_data_gen)

# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# plotImages(sample_training_images[:5])
#
# The model consists of three convolution blocks with a max pool layer in each of them.
# There's a fully connected layer with 512 units on top of it that is activated by a relu activation function.
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)),
    MaxPooling2D(2,2),
    #Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    #Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.summary()

def train():
    history = model.fit(
        x=train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE,
        callbacks=[cp_callback]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def load(filenames):
    from PIL import Image
    from skimage import transform
    imgs = []
    for filename in filenames:
        img = Image.open(filename)
        img = np.array(img).astype('float32') / 255
        img = transform.resize(img, (IMG_HEIGHT, IMG_WIDTH, 3))
        plotImage(img)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

def visualize_features(model):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    
    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
    
    global train_cats_dir, train_dogs_dir, train_cat_fnames, train_dog_fnames
    # Let's prepare a random input image of a cat or dog from the training set.
    cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
    dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
    
    img_path = random.choice(cat_img_files + dog_img_files)
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    
    x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
    x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)
    
    # Rescale by 1/255
    x /= 255.0
    
    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)
   
    print('Feature maps:',successive_feature_maps)
    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    
    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
      
      if len(feature_map.shape) == 4:
        
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        
        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
          x  = feature_map[0, :, :, i]
          x -= x.mean()
          x /= x.std ()
          x *=  64
          x += 128
          x  = np.clip(x, 0, 255).astype('uint8')
          display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid
    
        #-----------------
        # Display the grid
        #-----------------
    
        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' )  
        plt.show()

if os.path.exists(checkpoint_name):
    model.load_weights(checkpoint_path)
    visual = True

    if visual:
        visualize_features(model)
    else:
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        test_imgs = ['cat1.jpg']

        test_imgs = [os.path.join('/tmp', fname) for fname in test_imgs]
        test_imgs = load(test_imgs)

        predictions = model.predict(test_imgs)
        print(predictions)

        for pred in predictions:
            pred = 0 if pred <= 0 else 1
            print(labels[pred])
else:
    train()
