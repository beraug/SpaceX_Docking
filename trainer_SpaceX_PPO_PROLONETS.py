#!/usr/bin/env python

#Base
import time
import random
from random import choice
import math
import datetime
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Gym
import gym
from gym import utils, spaces
from gym import wrappers
from gym.envs.registration import register

#Tensorflow
#Tensorflow
import tensorflow as tf 
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
from tensorboardX import SummaryWriter
from tensorflow.python.framework.ops import disable_eager_execution

#Other scripts
import SpaceX_PPO_PROLONETS as PPO_model
from worker_SpaceX_PPO_PROLONETS import Worker
from Config_data_SpaceX import Config as Config

# import our training environment
import gym_spacex.envs.SpaceX_PPO_PROLONETS_env

disable_eager_execution()

env = gym.make("SpaceX_PPO_PROLONETS-v0")

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
summary_writer = SummaryWriter(Config.TRAINING_LOG_PATH)

chief = Worker('Chief', env, summary_writer, global_episodes)

#tf.global_variables_initializer()

saver = tf.train.Checkpoint(optimizer=tf.optimizers.Adam())
PPO_in = PPO_model.PPO()
PPO_in.load_model(saver)
chief.process()
