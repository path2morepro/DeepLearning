# IMPORTS
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

# Set seed from random number generator, for better comparisons
import numpy as np
from numpy.random import seed
seed(123)

import matplotlib.pyplot as plt

# from ray import train

# # Set seed from random number generator, for better comparisons
# from numpy.random import seed
# seed(123)

# import matplotlib.pyplot as plt

# define funstion that builds a CNN model
def build_CNN(input_shape, loss, 
                n_conv_layers:int=2, 
                n_filters:int=16, 
                n_dense_layers:int=0, 
                n_nodes:int=50, 
                use_dropout:bool=False, 
                learning_rate:float=0.01, 
                act_fun='relu', 
                optimizer:str='sgd',
                print_summary:bool=False,
                l2_regularization = False):
    """
    Builds a Convolutional Neural Network (CNN) model based on the provided parameters.
    
    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    loss (tf_keras.losses): Loss function to use in the model.
    n_conv_layers (int, optional): Number of convolutional layers in the model. Default is 2.
    n_filters (int, optional): Number of filters in each convolutional layer. Default is 16.
    n_dense_layers (int, optional): Number of dense layers in the model. Default is 0.
    n_nodes (int, optional): Number of nodes in each dense layer. Default is 50.
    use_dropout (bool, optional): Whether to use Dropout after each layer. Default is False.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
    act_fun (str, optional): Activation function to use in each layer. Default is 'sigmoid'.
    optimizer (str, optional): Optimizer to use in the model. Default is SGD.
    print_summary (bool, optional): Whether to print a summary of the model. Default is False.
    
    Returns:
    model (Sequential): Compiled Keras Sequential model.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------  

    # Setup optimizer, depending on input parameter string
    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    # ============================================
    
    # Setup a sequential model
    model = Sequential()

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------  
    # Add convolutional layers
    if l2_regularization:
        model.add(Conv2D(n_filters, 
                kernel_size=(3, 3), 
                activation=act_fun, 
                strides=(1, 1),
                use_bias=True,
                padding='same',
                kernel_regularizer=l2(0.03)))
        model.add(MaxPooling2D((2, 2)))
        for i in range(n_conv_layers-1):
            model.add(Conv2D(n_filters, 
                            kernel_size=(3, 3), 
                            activation=act_fun, 
                            strides=(1, 1),
                            use_bias=True,
                            padding='same',
                            kernel_regularizer=l2(0.03)))
            model.add(MaxPooling2D((2, 2)))

    else:
        model.add(Conv2D(n_filters, 
                                kernel_size=(3, 3), 
                                activation=act_fun, 
                                strides=(1, 1),
                                use_bias=True,
                                input_shape=input_shape,
                                padding='same'))
        model.add(MaxPooling2D((2, 2)))
        for i in range(n_conv_layers-1):
            model.add(Conv2D(n_filters, 
                            kernel_size=(3, 3), 
                            activation=act_fun, 
                            strides=(1, 1),
                            use_bias=True,
                            padding='same'))
            model.add(MaxPooling2D((2, 2)))
    
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    
    
    # Add dense layers
    for i in range(n_dense_layers):
        model.add(Dense(n_nodes, activation=act_fun))
        if use_dropout:
            model.add(Dropout(0.5))
    # I searched some information, it is said that
    # it is easy to overfit after dense layer

    # Add output layer
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=['accuracy'])

    # ============================================

    # Print model summary if requested
    if print_summary:
        model.summary()
    
    return model

# =======================================
# PLOTTING FUNCTIONS
# =======================================

# TRAINING CURVES PLOT FUNCTION
def plot_results(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.
    Parameters:
    history (keras.callbacks.History): A History object returned by the fit method of a Keras model. 
                                       It contains the training and validation loss and accuracy for each epoch.
    Returns:
    None
    """
    
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training','Validation'])

    plt.figure(figsize=(10,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training','Validation'])

    plt.show()


# =======================================
# AUGMENTATIONS FUNCTIONS
# =======================================

# ROTATE IMAGES BY () DEGREES
def myrotate(images):

    images_rot = np.rot90(images, axes=(1,2))
    
    return images_rot