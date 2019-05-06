# The function below converts a number between 1 and 70000 to the file name
# of the corresponding example
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


TRAIN_SIZE = 60000
TEST_SIZE = 10000
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


def num_to_ex_file_name(n):
    """Converts integer into the path of corresponding example image."""
    n = round(n)  # to ensure that n is an integer
    if n >= 1 and n <= TRAIN_SIZE:
        return "../train/"+str(n)+".png"
    elif n >= TRAIN_SIZE+1 and n <= TRAIN_SIZE+TEST_SIZE:
        return "../test/"+str(n)+".png"
    else:
        raise IndexError("Example ID is out of range")


def file_name_to_grayscale_image(filename, unroll=False):
    """Load grayscale image into a numpy array."""
    image_data = plt.imread(filename)
    # Since the images are grayscale,
    # this function loads data from only one channel.
    if unroll:
        return image_data[:, :, 0].ravel()
    return image_data[:, :, 0]

# Loads all training data into an appropriately formatted array


def load_train_data(m=TRAIN_SIZE, set_small_values_to_zero=False):
    """
    load_train_data(m=TRAIN_SIZE,set_small_values_to_zero=False)

    This function loads the train data from the train folder, returning
    the dataset X and labels y.

    Parameters
    ----------
    m : positive integer
        number of training examples to load
    set_small_values_to_zero : bool
        If True, we set pixels that are nearly black to be completely
        black. I doubt this will do much one way or the other,
        but because all the images are approximately zero near the
        boundary already, I thought that I would experiment
        to see if this avoids any weirdness that might result
        from filling in values during transformations
        used for dataset augmentation.

    Returns
    -------
    X: 4-D array with shape (m,1,IMAGE_HEIGHT,IMAGE_WIDTH)
        dataset in a channels_first format
        (with one channel as it is grayscale)
    y: 2-D array with shape (m,number_of_classes)
        labels for X
    """
    y = np.round(np.genfromtxt('../train.csv', delimiter=',')[1 : m+1, 1])
    samplefiles = np.vectorize(num_to_ex_file_name)(np.arange(1, m+1))
    X = np.zeros([m, 1, IMAGE_HEIGHT, IMAGE_WIDTH])
    for i in range(m):
        X[i][0] = file_name_to_grayscale_image(samplefiles[i])
    m, dummy, H, W = X.shape
    # Optionally set small values to zero w/o introucing discontinuity
    # NOTE: This should likely be moved to augmentation.py
    if set_small_values_to_zero:
        X[X < 0.025] = 0
        values = X[np.logical_and(X >= 0.025, X <= 0.1)]
        X[np.logical_and(X >= 0.025, X <= 0.1)
          ] = (1+np.tanh(values/(1 - values*values)))/2
    # Put y in appropriate format
    if not np.logical_or(y == 0, y == 1).all():
        y = to_categorical(y)
    return X, y
