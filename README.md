# Introduction

This is my work on a practice problem that can be found at https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels/. This is a very early phase of my project. I ultimately hope to carry out a detailed exploration of image classification methodology using this dataset and others. The code works and is documented, and the scripts `code/convscript_augmentinmemory.py` and `code/fcscript_augmentinmemory.py` can be run successfully.

# How to Use

I recommend using the IPython console when running scripts so that you can explore the results of each run. I am running my code using Keras 2.2.4 and Tensorflow 1.13.1 from Anaconda. It has been tested on my personal laptop (MacOS 10.14, on CPU), and on a remote server (Ubuntu 19.04, on GPU).

**In order for the models to build successfully at this time (5/5/2019) for training on a GPU, an updated version of `tensorflow_backend.py` must be downloaded from the Keras GitHub repository.**

This is due to a bug in Keras which has been resolved, but the fix for which has not yet been added to the Anaconda repository.

**The zip file containing the training data must be extracted before the data can be loaded.**

This is because it is much faster to download one zip file than 60000 small image files, so I left the training data folder out of the repository. In the future, I will replace the `train` folder with a single file that can be loaded more efficiently and does not require additional steps on the part of the user.

# Development

I am prioritizing documentation and reproducibility over rushing to tune hyperparameters and get on the leaderboard. The next step in my project is to create a pdf in which I will outline my strategy for testing and parameter tuning. After that, I will create code to compile, save, and visualize detailed information about test runs. I have other code that I have written and various tests that I have already run, but I am only adding fully documented material to the GitHub repository. There are not yet any documented test runs, but there will be.

The reader can infer what hyperparameters I'm likely to focus on by looking at the parameters of my functions, especially `augmentation.randomize_image` and the model construction functions in `code/modelbuilder.py`. I may, however, eliminate some of these parameters: my early testing will focus on which parameters are and aren't useful to explore.

# Workflow

This code is developed in Emacs on my personal laptop, and run primarily on my "server," a borrowed gaming laptop based in Idaho. I connect to the server via ssh over a VPN. The server works about 20 times faster during training due to its GPU. I generally work on the code and do small-scale tests on my laptop, and then sync the server's code with the GitHub repository before running more intensive tests using a combination of ssh, GNU Screen, and IPython.

# Notes

I am fully aware that my convolutional neural network model should outperform my fully connected neural network model. However, I am interested in whether this image set being centered and generally neat will mean that the advantages of convolutional neural networks will be smaller than they would have been with a "messier" dataset.

I am also aware that Keras has built-in support for image augmentation. I have tested `keras.preprocessing.ImageDataGenerator` (without documentation as of yet) and am going to compare the performance of this option to in-memory augmentation during the testing process.
