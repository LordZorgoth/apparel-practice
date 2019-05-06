This is a work in progress. The code works and is documented, and the scripts `code/convscript_augmentinmemory.py` and `code/fcscript_augmentinmemory.py` can be run. However, no documented testing has been run to tune parameters. The next step is to create a pdf in which I will outline my testing strategy, and to create code to obtain and visualize information about test runs. I have other code that I have written and various tests that I have already run, but I am only adding fully documented material to the GitHub repository.

I am running my code using Keras 2.2.4 and Tensorflow 1.13.1 from Anaconda. *In order for the models to build successfully at this time (5/5/2019), an updated version of `tensorflow_backend.py` must be downloaded from the Keras GitHub.* This is due to a bug in Keras which has been resolved, but the fix for which has not been added to the Anaconda repository.

This code has been tested on my personal laptop, using MacOS 10.14, and on my "server", an old gaming laptop running Ubuntu 19.04 belonging to my Dad based in Idaho. It runs about 20 times faster on the gaming laptop, as it has an NVIDIA GPU while my MacBook does not.

I am fully aware that the convolutional neural network should easily outperform the fully connected neural network. I am including the fully connected model to get a better idea of how both types of networks work, and to see how great the performance advantage of convolutional neural networks is for a given level of computing power.

The reader can infer what parameters I'm likely to try to vary by looking at the parameters of many of my functions, especially `augmentation.randomize_image` and the model building functions in `code/modelbuilder.py`

I am aware that Keras has built-in support for image augmentation.
I have tested `keras.preprocessing.ImageDataGenerator` (without documentation)
and am going to compare the performance of this option to in-memory augmentation.
I chose to use a custom preprocessing function rather than the builtin features
of the `ImageDataGenerator` class because I feel that I need more fine-grained
control in order to learn what works and what doesn't. Fortunately, I don't
seem to take any significant performance hit with my preprocessing function.