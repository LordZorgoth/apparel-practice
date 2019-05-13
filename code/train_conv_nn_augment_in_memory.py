import numpy as np
from load_data import load_train_data
from model_building import model_build_conv
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from keras.metrics import categorical_accuracy
from utils import convert_arrays_to_accuracy, TimeHistory
import augmentation as aug

# m is the number of examples to load from the training dataset.
# Reducing m is especially useful when debugging to allow
# rapid training runs. For the full dataset, use m=60000.
m = 60000
epochs = 150
# n_transforms is the number of transformations to create for each image
n_transforms = 10

X, y = load_train_data(m)

# Ensure that we always use the same training and cross-validation sets
# by always using 1 as the seed for the PRNG.
np.random.seed(1)
Xtr, Xval, ytr, yval = train_test_split(X, y, train_size=0.6, test_size=0.4)
Xtr, ytr = aug.augment_dataset(Xtr, ytr, n_transforms, fixed_seeds=True)

model = model_build_conv()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=[categorical_accuracy])
my_hist = History()
time_hist = TimeHistory()
# Because we are feeding in an augmented dataset using model.fit,
# the training categorical accuracy returned by the model will be
# based on the augmented training set. However, we are far more
# interested in the true training accuracy, based on predictions
# from the unaugmented training set. The for loop below is an ugly hack.
# I will replace it with a callback at some point.
train_accs = []
for E in range(epochs):
    model.fit(Xtr, ytr, verbose=2, callbacks=[my_hist, time_hist],
              validation_data=(Xval, yval), epochs=E+1, initial_epoch=E,
              batch_size=32, shuffle=True)
    train_acc = convert_arrays_to_accuracy(
                    model.predict(Xtr[0:: n_transforms+1]),
                    ytr[0:: n_transforms+1]
                    )
    train_accs.append(train_acc)
    print("True training accuracy is "+str(train_acc)[0:6]+".\n")
