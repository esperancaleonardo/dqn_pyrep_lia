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
from Models import *



class Agent(object):

    def __init__(self, memory_size, batch_size, input_dimension, number_of_actions, alpha, load_weights, model_string, file=""):
        super(Agent, self).__init__()
        self.memory = deque(maxlen=memory_size)
        metrics = ['accuracy', 'mean_squared_error']
        self.model = self.select_model(model_string, input_dimension, number_of_actions, 'mean_squared_error', Adam(lr=alpha),  metrics)
        self.number_of_actions = number_of_actions
        self.input_dimension = input_dimension
        self.model_string = model_string
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
                                    # self.model.predict([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
                                    #                     state[1].reshape(1,self.input_dimension,self.input_dimension,1),
                                    #                     state[2].reshape(1,self.input_dimension,self.input_dimension,1)])[0]
                                    self.model.predict([state[1].reshape(1,self.input_dimension,self.input_dimension,1)])[0]
                                                 )
                                         )
                         )

            # target_f = self.model.predict([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
            #                                state[1].reshape(1,self.input_dimension,self.input_dimension,1),
            #                                state[2].reshape(1,self.input_dimension,self.input_dimension,1)])

            target_f = self.model.predict([state[1].reshape(1,self.input_dimension,self.input_dimension,1)])

            target_f[0][action] = target
            # fit = self.model.fit([state[0].reshape(1,self.input_dimension,self.input_dimension,1),
            #                       state[1].reshape(1,self.input_dimension,self.input_dimension,1),
            #                       state[2].reshape(1,self.input_dimension,self.input_dimension,1)],
            #                       target_f, epochs, verbose=0)

            fit = self.model.fit([state[1].reshape(1,self.input_dimension,self.input_dimension,1)],
                                  target_f, epochs, verbose=0)


        if fit == None:
            return 0
        else:
            return fit

    def act(self, state, epsilon):

        if random.uniform(0,1) <= epsilon:
            return np.random.randint(0,self.number_of_actions)
        else:
            state1 = np.array(state[0])
            state2 = np.array(state[1])
            state3 = np.array(state[2])
            # action_values = self.model.predict([state1.reshape(1,self.input_dimension,self.input_dimension,1),
            #                                     state2.reshape(1,self.input_dimension,self.input_dimension,1),
            #                                     state3.reshape(1,self.input_dimension,self.input_dimension,1)])

            action_values = self.model.predict([state2.reshape(1,self.input_dimension,self.input_dimension,1)])

        return np.argmax(action_values[0])


    def action_to_vel(self, action):
        vell = [0.0 for i in range(7)]
        if action%2 == 0:   vell[action % 7] += 1.0
        else:               vell[action % 7] -= 1.0

        return vell

    def select_model(self, model_string, input_dimension, number_of_actions, loss_type, optimizer, metrics_list):

        if model_string == '3_input':
            return tree_input_cnn(input_dimension, number_of_actions, loss_type, optimizer, metrics_list)
        elif model_string == '1_input':
            return single_input_cnn(input_dimension, number_of_actions, loss_type, optimizer, metrics_list)
        elif model_string == 'base':
            return model_paper_cnn(input_dimension, number_of_actions, loss_type, optimizer, metrics_list)
