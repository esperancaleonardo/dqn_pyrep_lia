import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from collections import deque
import cv2 as cv, os
import numpy as np
from time import sleep
import math, random
from tqdm import tqdm
from os.path import dirname, join, abspath



class Agent(object):

    def __init__(self, memory_size, batch_size, input_dimension, number_of_actions, alpha, load_weights, file=""):
        super(Agent, self).__init__()
        self.memory = deque(maxlen=memory_size)
        metrics = ['accuracy', 'mean_squared_error']
        self.model = self.create_model(input_dimension, number_of_actions, 'mean_squared_error', Adam(lr=alpha),  metrics)
        self.number_of_actions = number_of_actions
        self.input_dimension = input_dimension
        self.BATCH_SIZE = batch_size
        self.STEP_SPEED = 10.0
        if load_weights and (file != ""):
            print("model will load now...")
            self.model.load_weights(join(dirname(abspath(__file__)),file))

    def write_memory(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def replay(self, gamma, epochs):
        mini_batch = random.sample(self.memory, int(self.BATCH_SIZE))
        fit = None
        for state, action, reward, done, next_state in tqdm(mini_batch):
            target = reward
            if not done:
                target = (reward + gamma*(np.amax(
                                    self.model.predict([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
                                                        state[1].reshape(1,self.input_dimension,self.input_dimension,1),
                                                        state[2].reshape(1,self.input_dimension,self.input_dimension,1)])[0]
                                                 )
                                         )
                         )

            target_f = self.model.predict([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
                                           state[1].reshape(1,self.input_dimension,self.input_dimension,1),
                                           state[2].reshape(1,self.input_dimension,self.input_dimension,1)])

            target_f[0][action] = target
            fit = self.model.fit([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
                                  state[1].reshape(1,self.input_dimension,self.input_dimension,1),
                                  state[2].reshape(1,self.input_dimension,self.input_dimension,1)],
                                  target_f, epochs, verbose=0)

        if fit == None:
            return 0
        else:
            return fit

    def act(self, state, epsilon):
        if np.random.randint(0,10) <= epsilon:
            return np.random.randint(0,self.number_of_actions)
        else:
            state1 = np.array(state[0])
            state2 = np.array(state[1])
            state3 = np.array(state[2])
            action_values = self.model.predict([state1.reshape(1,self.input_dimension,self.input_dimension,1),
                                                state2.reshape(1,self.input_dimension,self.input_dimension,1),
                                                state3.reshape(1,self.input_dimension,self.input_dimension,1)])
        return np.argmax(action_values[0])


    def create_model(self, input_dimension, number_of_actions, loss_type, optimizer, metrics_list):
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

        x = concatenate([model1.output, model1.output, model1.output])
        x = Dense(4096)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(number_of_actions)(x)

        model = Model(inputs=[model1.input, model2.input, model3.input], outputs=x)
        model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)
        return model
