import keras.layers as lyr
import keras.models as mdl
import keras.regularizers as reg
import numpy as np


def model_build_conv(conv_layer_count=[1, 2], conv_vs_pool=True,
                     init_filters=32, lambda_conv=0, lambda_dense=0.001,
                     global_average_vs_max=False, final_pool_size=3,
                     dense_layers=[50, 10], dropout=[0.5, 0.25]):
    """ 
    model_build_conv(conv_layer_count=[1, 2], conv_vs_pool=True,
                     init_filters=32, lambda_conv=0, lambda_dense=0.001,
                     global_average_vs_max=False, final_pool_size=3,
                     dense_layers=[50, 10], dropout=[0.5, 0.25])

    This builds the Keras model for our convolutional neural network.
    We start with convolutional layers consisting of 5x5 filters
    operating on 28x28 images, then we map to to 14x14 images, either
    using max pooling or a convolutional layer with a stride of 2.
    We then apply convolutional layers consisting of 3x3 filters,
    with the last layer having no padding, resulting in 12x12 images.
    We then apply either global average pooling or max pooling before
    feeding our features into a dense network.

    The convolutional layers are regularized with batch normalization,
    while the dense layers are regularized with dropout.

    Parameters
    ----------
    conv_layer_count : list of two positive integers
        conv_layer_count[0] is the number of convolutional layers
        to apply before reducing from 28x28 images to 14x14,
        while conv_layer_count[1] is the number of convolutional
        layers to apply to 14x14 images before the final pooling step.
    conv_vs_pool : bool
        If conv_vs_pool is True, we will use a convolutional layer
        with 5x5 filters and a stride of 2 to reduce from 28x28 to
        14x14 images. If False, we will use max pooling with a
        pool_size of 2 instead.
    init_filters : positive integer
        the number of filters to start with; we will increase the number
        of filters in powers of 2 as our features become smaller.
    lambda_conv : nonnegative float
        L2 regularization parameter for convolutional layers; as we
        are already regularizing with batch normalization, we will
        most likely set this to zero.
    lambda_dense : nonnegative float
        L2 regularization parameter for dense layers.
    global_average_vs_max : bool
        determines whether our final pooling layer is a global average
        or max pooling layer
    final_pool_size : positive integer
        If global_average_vs_max is False, this variable determines
        pool_size for the final max pooling layer.
    dense_layers : list of positive integers
        List of sizes of dense layers. The last element
        is the output layer.
        dense_layers[-1] should be the number of classes.
    dropout : list of floats between 0 and 1
        list of dropout probabilities for dense layers

    Returns
    -------
    model : keras.engine.sequential.Sequential
        the resultant Keras model
    """
    model = mdl.Sequential()
    # Start with conv_layer_count[0] convolutional layers operating
    # on 28x28 images
    for i in range(conv_layer_count[0]):
        model.add(
            lyr.Conv2D(
                init_filters, (5, 5),
                input_shape=(1, 28, 28), strides=(1, 1),
                data_format='channels_first', padding='same',
                kernel_regularizer=reg.l2(lambda_conv))
            )
        model.add(lyr.BatchNormalization(axis=1))
        model.add(lyr.Activation('relu'))
    # Reduce dimensions of images from 28x28 to 14x14, either via a
    # convolutional layer or a max pooling layer.
    if conv_vs_pool:
        model.add(
            lyr.Conv2D(2*init_filters, (5, 5),
                       input_shape=(1, 28, 28), strides=(2, 2),
                       data_format='channels_first', padding='same',
                       kernel_regularizer=reg.l2(lambda_conv))
            )
        model.add(lyr.BatchNormalization(axis=1))
        model.add(lyr.Activation('relu'))
    else:
        model.add(lyr.MaxPool2D(pool_size=(2, 2),
                                data_format='channels_first'))
    # We now add conv_layer_count[1] convolutional layers operating
    # on 14x14 images. The last layer outputs a 12x12 image.
    padding_list = ['same']*(conv_layer_count[1]-1) + ['valid']
    for i in range(conv_layer_count[1]):
        model.add(
            lyr.Conv2D(4*init_filters, (3, 3),
                       input_shape=(1, 14, 14), strides=(1, 1),
                       data_format='channels_first',
                       padding=padding_list[i],
                       kernel_regularizer=reg.l2(lambda_conv))
            )
        model.add(lyr.BatchNormalization(axis=1))
        model.add(lyr.Activation('relu'))
    # We now reduce our 12x12 images to produce the final output of the
    # convolutional portion of the network,
    # either through global average pooling or max pooling.
    if global_average_vs_max:
        model.add(lyr.GlobalAveragePooling2D(data_format='channels_first'))
        input_size = 4*init_filters
    else:
        model.add(
            lyr.MaxPool2D(pool_size=(final_pool_size, final_pool_size),
                          data_format='channels_first')
            )
        input_size = 4 * init_filters * (12/final_pool_size)**2
    # Final stage of network: dense layers.
    model.add(lyr.Flatten(data_format='channels_first'))
    activation_list = ['relu']*(len(dense_layers)-1) + ['softmax']
    for i in range(len(dense_layers)):
        model.add(lyr.Dropout(dropout[i]))
        model.add(
            lyr.Dense(dense_layers[i], input_shape=(input_size,),
                      kernel_regularizer=reg.l2(lambda_dense))
            )
        model.add(lyr.Activation(activation_list[i]))
        input_size = dense_layers[i]
    return model


def model_build_dense(lambda_dense=0.00002, input_shape=(1, 28, 28),
                   layers=[400, 250, 100, 50, 10],
                   dropout=[0.25, 0.5, 0.5, 0.5, 0.25]):
    """
    model_build_dense(lambda_dense=0.00002, input_shape=(1, 28, 28),
                   layers=[400, 250, 100, 50, 10],
                   dropout=[0.25, 0.5, 0.5, 0.5, 0.25])

    Builds a Keras model for a dense neural network with
    dropout and L2 regularization.

    Parameters
    ----------
    lambda_dense : nonnegative float
        L2 regularization parameter
    input_shape : tuple of positive integers
        shape of input
    layers : list of positive integers
        size of all layers from the first hidden layer
        to the output layer
    dropout : list of floats between 0 and 1
        list of dropout probabilities to apply before each layer

    Returns
    -------
    model : keras.engine.sequential.Sequential
        the resultant Keras model
    """
    model = mdl.Sequential()
    model.add(lyr.Flatten(data_format='channels_first',
                          input_shape=input_shape)
             )
    input_size = np.product(input_shape)
    activation_list = ['relu']*(len(layers)-1) + ['softmax']
    for i in range(len(layers)):
        model.add(lyr.Dropout(dropout[i]))
        model.add(
            lyr.Dense(layers[i], input_shape=(input_size,),
                            kernel_regularizer=reg.l2(lambda_dense))
            )
        model.add(lyr.Activation(activation_list[i]))
        input_size = layers[i]
    return model
