#!/usr/bin/env python

#Base
import os
import random
import numpy as np
import datetime

#Tensorflow
import tensorflow as tf 
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras import backend as K
from keras.optimizers import Adam
import numba as nb
from tensorboardX import SummaryWriter

#Other scripts
from Config_data_SpaceX import Config as Config


class PPO:

    def __init__(self):

        self.num_state = 6
        self.num_action = 7
        self.cliprange = Config.CLIP_RANGE
        self.entropy_loss = Config.ENTROPY_LOSS
        self.checkpoint_path = Config.TRAINING_LOG_PATH
        self.lr_A1 = Config.LEARNING_RATE_A1
        self.lr_A2 = Config.LEARNING_RATE_A2
        self.lr_A3 = Config.LEARNING_RATE_A3
        self.lr_C = Config.LEARNING_RATE_C
        self.epochs = Config.EPOCHS
        self.hidden_size = Config.HIDDEN_SIZE
        self.num_layers = Config.NUM_LAYERS
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.environment = Config.ENV_NAME

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value= 1 - self.cliprange, max_value= 1 + self.cliprange) * advantage) + self.entropy_loss * -(prob * K.log(prob + 1e-10)))
        return loss

    def load_model(self, saver):
        
        checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
        print(checkpoint)
        if checkpoint:
            saver.restore(checkpoint)
            print('.............Model restored to global.............')
        else:
            print('................No model is found.................')

    def save_model(self, actor, time_step):

        print('............save model ............')
        checkpoint_name = self.checkpoint_path + self.environment + "_" + str(time_step) + ".ckpt"
        print(checkpoint_name)
        actor.save_weights(checkpoint_name)
        print('............model saved ............')
        
    def build_critic_net(self):

        state_input = Input(shape=(self.num_state,))
        x = Dense(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_size, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.lr_C), loss='mse')

        return model

    def build_actor_net(self, learning_rate):
        
        state_input = Input(shape=(self.num_state,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.num_action,))

        x = Dense(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_size, activation='tanh')(x)

        out_actions = Dense(self.num_action, activation='softmax', name='output', dtype='float32')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=learning_rate), loss=[self.proximal_policy_optimization_loss(advantage,old_prediction)])

        model.summary()

        return model


    def build_actor_lstm_net(self, learning_rate):
        
        state_input = Input(shape=(10, self.num_state,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(10, self.num_action,))

        x = LSTM(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = LSTM(self.hidden_size, activation='tanh')(x)

        out_actions = Dense(self.num_action, activation='softmax', name='output', dtype='float32')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=learning_rate), loss=[self.proximal_policy_optimization_loss(advantage,old_prediction)])

        model.summary()

        return model

    def build_critic_lstm_net(self):

        state_input = Input(shape=(10, self.num_state,))
        x = LSTM(self.hidden_size, activation='tanh')(state_input)
        for _ in range(self.num_layers - 1):
            x = LSTM(self.hidden_size, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.lr_C), loss='mse')

        return model