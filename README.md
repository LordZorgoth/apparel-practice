This is a work in progress. The code works and is documented, and the scripts `code/convscript_augmentinmemory.py` and `code/fcscript_augmentinmemory.py` can be run successfully. I recommend using the IPython console so that you can explore the results of each run.

I am prioritizing documentation and replicability over rushing to testing. The next step in my project is to create a pdf in which I will outline my testing strategy, and to create code to obtain and visualize information about test runs. I have other code that I have written and various tests that I have already run, but I am only adding fully documented material to the GitHub repository. There are not yet any documented test runs, but there will be.

The reader can infer what parameters I'm likely to try to vary by looking at the parameters of my functions, especially `augmentation.randomize_image` and the model contstruction functions in `code/modelbuilder.py`

I am running my code using Keras 2.2.4 and Tensorflow 1.13.1 from Anaconda. **In order for the models to build successfully at this time (5/5/2019), an updated version of `tensorflow_backend.py` must be downloaded from the Keras GitHub.** This is due to a bug in Keras which has been resolved, but the fix for which has not been added to the Anaconda repository. Second, **the zip file containing training data must be extracted**. This is because it is much faster to download one zip file than 60000 small image files, so I left the training data folder out of the repository. In the future, I will replace the train folder with a single file that can be loaded more efficiently and does not require additional steps on the part of the user after cloning the repository.

This code has been tested on my personal laptop, using MacOS 10.14, and on my "server," an old gaming laptop I borrowed, which is running Ubuntu 19.04 and is based in Idaho. I connect to the gaming laptop via ssh over a VPN. It trains models about 20 times faster than my MacBook due to its NVIDIA GPU. My workflow is generally to write the code and do small-scale tests in the ipython console on my laptop, and then to update the server's code through `git` and run larger tests using a combination of ssh, GNU screen, and IPython.

I am fully aware that the convolutional neural network should easily outperform the fully connected neural network. I am including the fully connected model to get a better idea of how both types of networks work, and to see how great the performance advantage of convolutional neural networks is for a given level of computing power.

I am also aware that Keras has built-in support for image augmentation.
I have tested `keras.preprocessing.ImageDataGenerator` (without documentation)
and am going to compare the performance of this option to in-memory augmentation.
I chose to use a custom preprocessing function rather than the builtin features
of the `ImageDataGenerator` class because I feel that I need more fine-grained
control in order to learn what works and what doesn't. Fortunately, my preprocessing function doesn't seem to cause a significant performance hit.