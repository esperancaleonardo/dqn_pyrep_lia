#####
#####
##### MODEL DEFINITIONS
#####
#####

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, concatenate, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

################################################################################

def single_input_cnn(input_dimension, number_of_actions, loss_type, optimizer,  metrics_list):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(256, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_actions, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)
    #model.summary()

    return model


def model_paper_cnn(input_dimension, number_of_actions, loss_type, optimizer,  metrics_list):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(256, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(number_of_actions, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)
    #model.summary()

    return model



def tree_input_cnn(input_dimension, number_of_actions, loss_type, optimizer,  metrics_list):

    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()

    model1.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Conv2D(8, (5, 5), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Conv2D(8, (5, 5), activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    model1.add(Dropout(0.5))
    model1.add(Flatten())
    ###################################################################################################
    model2.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(8, (5, 5), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(8, (5, 5), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.5))
    model2.add(Flatten())
    ###################################################################################################
    model3.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Conv2D(8, (5, 5), activation='relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Conv2D(8, (5, 5), activation='relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Dropout(0.5))
    model3.add(Flatten())

    model = concatenate([model1.output, model2.output, model3.output])
    model = Dense(4096, kernel_initializer='random_uniform', bias_initializer='zeros')(model)
    model = Dense(256, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')(model)
    model = Dropout(0.2)(model)
    x = Dense(number_of_actions,  kernel_initializer='random_uniform', bias_initializer='zeros')(model)

    model = Model(inputs=[model1.input, model2.input, model3.input], outputs=x)
    model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)
    #model.summary()

    return model
