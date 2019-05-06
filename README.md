This is my work on a practice problem that can be found at https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels/. This is a very early phase of my project, in which I hope to carry out a detailed exploration of image classification methodology. The code works and is documented, and the scripts `code/convscript_augmentinmemory.py` and `code/fcscript_augmentinmemory.py` can be run successfully. I recommend using the IPython console so that you can explore the results of each run.

I am running my code using Keras 2.2.4 and Tensorflow 1.13.1 from Anaconda.

**In order for the models to build successfully at this time (5/5/2019) for training on a GPU, an updated version of `tensorflow_backend.py` must be downloaded from the Keras GitHub repository.**

This is due to a bug in Keras which has been resolved, but the fix for which has not been added to the Anaconda repository.

**The zip file containing the training data must be extracted.**

This is because it is much faster to download one zip file than 60000 small image files, so I left the training data folder out of the repository. In the future, I will replace the zip corresponding to the `train` folder with a single file that can be loaded more efficiently and does not require additional steps on the part of the user.

I am prioritizing documentation and replicability over rushing to tune hyperparameters and get on the leaderboard. The next step in my project is to create a pdf in which I will outline my testing strategy, and to create code to obtain and visualize information about test runs. When I have done this, I will drastically shorten this README. I have other code that I have written and various tests that I have already run, but I am only adding fully documented material to the GitHub repository. There are not yet any documented test runs, but there will be.

The reader can infer what hyperparameters I'm likely to focus on by looking at the parameters of my functions, especially `augmentation.randomize_image` and the model construction functions in `code/modelbuilder.py`. I may, however, eliminate some of these parameters: my early testing is likely to focus on which of my parameters are and aren't useful to explore.

This code has been tested on my personal laptop, using MacOS 10.14, and on my "server," a borrowed gaming laptop, which is running Ubuntu 19.04 and is based in Idaho. I connect to the server via ssh over a VPN. The server works about 20 times faster during training due to its GPU. My workflow is generally to first update the code and do small-scale tests on my laptop, and then to update the server's code using git and run larger tests using a combination of ssh, GNU screen, and IPython.

I am fully aware that my convolutional neural network model should outperform my fully connected neural network model. However, I am interested in whether this image set being centered and generally neat will mean that the advantages of convolutional neural networks will be smaller than they would have been with a "messier" dataset.

I am also aware that Keras has built-in support for image augmentation. I have tested `keras.preprocessing.ImageDataGenerator` (without documentation) and am going to compare the performance of this option to in-memory augmentation during the testing process.