import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as img

def randomize_image(image,max_shift=2,p_same=0,fixed_seeds=False,seed_count=10,
                    seed_index=None,p_rotate=0.75,max_angle=10,p_flip=0.2):
    """
    randomize_image(image,max_shift=2,p_same=0,fixed_seeds=False,seed_count=10,
                    seed_index=None,p_rotate=0.75,max_angle=10,p_flip=0.2)

    Returns augmentation of dataset X and labels y. This augmentation
    is produced by adding random transformations of each element of X.

    Parameters
    ----------
    image : 3-D array of size (1, height, width)
        image to be processed
    p_same : float between 0 and 1
        probability of a given "transformation" being forced to be
        identical to the original
    p_rotate : float between 0 and 1
        probability of applying a random rotation
    p_flip : float between 0 and 1
        probability of reflecting the image horizontally
    max_angle : nonnegative number < 360
        maximum rotation angle, in degrees
    max_shift : nonnegative integer
        maximum number of pixels by which to translate in each direction
    fixed_seeds : bool
        If fixed_seeds is True, "random" seeds will be deterministically
        computed for reproducibility, and there will be a set list of seeds
        used so that only a certain number of image transformations are
        possible.
    seed_count : positive integer
        If fixed_seeds is True, this is the number of image transformations
        for each image.
    seed_index : None or integer in range(0,seed_count)
        If fixed_seeds is True seed_index is None, a transformation will
        be selected from the list of possible transformations at random.
        If fixed_seeds is True and seed_index is not None, it selects a
        specific element from the set of possible transformations. This 
        is used in the function augment_dataset to ensure that we get
        one of each transformation.

    Returns
    -------
    image : the transformed image
    
    """
    #Our images have channels first, so height is shape[1] and width is shape[2]
    H,W=image.shape[1],image.shape[2]
    #Return image unchanged with probability p_same.
    np.random.seed()
    if np.random.rand()<p_same:
        return image
    #If fixed_seeds is True, we are ensuring that there are only seed_count
    #possible transformations of image
    if fixed_seeds:
        #Generate a unique value for each image by taking a subarray of
        #6 "random" non-zero, non-one values and hashing its string;
        #we will use this value to create a fixed but unique list of
        #random seeds for each image
        np.random.seed(1)
        np.random.seed(hash(
            np.random.choice(image[np.logical_and(image!=0,image!=1)],6)
            .tostring())%2**32)
        #generate a list of seed_count seeds, still uniquely corresponding
        #to the image
        seed_choices=np.random.randint(2**32,size=seed_count)
        #Select one of the seeds in seed_choices
        if seed_index is None:
            np.random.seed()
            np.random.seed(seed_choices[np.random.randint(seed_count)])
        else:
            np.random.seed(seed_choices[seed_index])
    #Rotate
    if np.random.rand()<p_rotate:
        angle=2*max_angle*(0.5-np.random.rand())
        image=img.rotate(image, angle, axes=(1,2),
                         reshape=False, output=None, order=2,
                         mode='nearest')
    #Reflect horizontally
    if np.random.rand()<p_flip:
            image=np.flip(image,2)
    #Translate
    h=np.random.randint(-max_shift,max_shift+1)
    w=np.random.randint(-max_shift,max_shift+1)
    image=img.shift(image,(0,h,w),order=0,mode='nearest')
    return image

def augment_dataset(X,y,n_copies,
                     max_shift=2,p_same=0,fixed_seeds=False,
                     p_rotate=0.75,max_angle=10,p_flip=0.2):
    """
    augment_dataset(X,y,n_copies,
                     max_shift=2,p_same=0,fixed_seeds=True,
                     p_rotate=0.75,max_angle=10,p_flip=0.2)

    Returns augmentation of dataset X and labels y. This augmentation
    is produced by adding random transformations of each element of X.

    Parameters
    ----------
    X : 4-D array of size (number of samples, 1, image height, image width)
        The data set to augment
    y : 2-D array of size (number of samples, number of classes)
        The labels corresponding to X
    n_copies : nonnegative integer
        The number of transformations of each image in X to create
    p_same : float between 0 and 1
        probability of a given "transformation" being forced to be
        identical to the original
    p_rotate : float between 0 and 1
        probability of applying a random rotation
    p_flip : float betewwn 0 and 1
        probability of reflecting the image horizontally
    max_angle : nonnegative number < 360
        maximum rotation angle, in degrees
    max_shift : nonnegative integer
        maximum number of pixels by which to translate in each direction
    fixed_seeds : bool
        If fixed_seeds is True, the augmented data set will always contain
        the same transformations across multiple runs of augment_dataset.

    Returns
    -------
    Xnew : 4-D array
        The augmented dataset
    ynew : 2-D array
        Labels for Xnew
    
    """
    m=X.shape[0]
    n=n_copies+1
    Xnew=np.tile(np.zeros_like(X),(n,1,1,1))
    ynew=np.tile(np.zeros_like(y),(n,1))
    for i in range(0,m):
        Xnew[i*n]=X[i]
        ynew[i*n]=y[i]
        for j in range(1,n):
            Xnew[i*n+j]=randomize_image(
                X[i],fixed_seeds=fixed_seeds,seed_count=n_copies,seed_index=j-1)
            ynew[i*n+j]=y[i]
    return Xnew,ynew
