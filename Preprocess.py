"""
Some codes from  https://github.com/Newmu/dcgan_code
and  make it work on Facadas datasets
"""
import scipy.misc
import numpy as np

# load_data for pix2pix
# A,B represents the two part of the image
# original picture is 512* 256 [CMP Facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)


def load_data(image_path, flip=True, is_test=False):
    A, B = load_image(image_path)
    A, B = preprocess_A_and_B(A, B, flip=flip, is_test=is_test)

    # cast to [-1: 1]
    A = A/127.5 - 1.
    B = B/127.5 - 1.

    AB = np.concatenate((A, B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    #0 1 2 in this function concatenate the RGB layer
    return AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    # fetch the size of the image
    # specialized for the dataset 512*256
    w2 = int(w/2)
    A = input_img[:, 0:w2]
    B = input_img[:, w2:w]
    return A, B


def preprocess_A_and_B(A, B, load_size=256, fine_size=240, flip=True, is_test=False):
    # resize the picture to 240
    # with the start from random place between 240-256
    if is_test:
        A = scipy.misc.imresize(A, [fine_size, fine_size])
        B = scipy.misc.imresize(B, [fine_size, fine_size])
    else:
        A = scipy.misc.imresize(A, [load_size, load_size])
        B = scipy.misc.imresize(B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        A = A[h1:h1 + fine_size, w1:w1 + fine_size]
        B = B[h1:h1 + fine_size, w1:w1 + fine_size]

        if flip and np.random.random() > 0.5:
            A = np.fliplr(A)
            B = np.fliplr(B)

    return A, B

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

# ----------

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.
