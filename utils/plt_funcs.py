# @Author: Michel Andy <ragnar>
# @Date:   2020-01-24T14:49:18-05:00
# @Email:  Andymic12@gmail.com
# @Filename: util.py
# @Last modified by:   ragnar
# @Last modified time: 2020-01-24T14:58:59-05:00
import os
import matplotlib.pyplot as plt

def save_fig(fig_id, tight_layout=True, dir='.'):
    path = os.path.join(dir, "images")

    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, fig_id + ".png")
    
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")
