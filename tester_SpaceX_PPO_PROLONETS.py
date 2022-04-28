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
import tensorflow as tf 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#Other scripts
from Config_data_SpaceX import Config as Config

# import our training environment
import gym_spacex.envs.SpaceX_PPO_PROLONETS_env

def load_model(actor, path):
        
    #saver = tf.train.Checkpoint(optimizer=tf.optimizers.Adam()) 
    checkpoint = tf.train.latest_checkpoint(path)
    print(checkpoint)
    if checkpoint:
        actor.load_weights(checkpoint)
        #saver.restore(checkpoint)
        print('.............Model restored to global.............')
    else:
        print('................No model is found.................')

def build_net():

    model = Sequential()
    model.add(Dense(8, input_dim=1))
    model.add(Dense(8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
        
    model.summary()

    return model

def pick_actor(obs, init_episodes):
        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = obs

        if np.linalg.norm(yaw) > 0.2:
            actor_num = 1
        elif np.linalg.norm(yaw) <= 0.2 and np.linalg.norm(pitch) > 0.2:
            actor_num = 2
        elif np.linalg.norm(yaw) <= 0.2 and np.linalg.norm(pitch) <= 0.2 and np.linalg.norm(roll) > 0.2:
            actor_num = 3
        elif np.linalg.norm(yaw) <= 0.2 and np.linalg.norm(pitch) <= 0.2 and np.linalg.norm(roll) <= 0.2 and np.linalg.norm(y) > 0.2:
            actor_num = 4
        elif np.linalg.norm(yaw) <= 0.2 and np.linalg.norm(pitch) <= 0.2 and np.linalg.norm(roll) <= 0.2 and np.linalg.norm(y) <= 0.2 and np.linalg.norm(z) > 0.2:
            actor_num = 5
        elif np.linalg.norm(yaw) <= 0.2 and np.linalg.norm(pitch) <= 0.2 and np.linalg.norm(roll) <= 0.2 and np.linalg.norm(y) <= 0.2 and np.linalg.norm(z) <= 0.2 and np.linalg.norm(x) > 0.0:
            actor_num = 6
        else:
            actor_num = 6

        return actor_num

def process():

        print("Starting the tester script")      
        total_episodes = 30
        obs = env.reset()
        x_t1 = []
        total_rewards = 0
        current_episode = 0
        episode_length = 0
        total_steps = 0
        val = True
        summaries = tf.compat.v1.summary.merge_all()
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

        while True:
            
            for i in range(0, total_episodes):

                states_buf = [np.zeros((1, 12), dtype=np.float32)]
                rewards_buf = [0]

                actor_num = pick_actor(obs, current_episode)
                if actor_num == 1:
                    x_t = obs[6]
                    print("ROT OBS IS: {0}".format(x_t))
                elif actor_num == 2:
                    x_t = obs[7]
                    print("ROT OBS IS: {0}".format(x_t))
                elif actor_num == 3:
                    x_t = obs[8]
                    print("ROT OBS IS: {0}".format(x_t))
                elif actor_num == 4:
                    x_t = obs[1]
                    print("TRANS OBS IS: {0}".format(x_t))
                elif actor_num == 5:
                    x_t = obs[2]
                    print("TRANS OBS IS: {0}".format(x_t))
                elif actor_num == 6:
                    x_t = obs[0]
                    print("TRANS OBS IS: {0}".format(x_t))

                action_obs = [[x_t]] 
                print(action_obs)
                action_pred = actor.predict(action_obs)
                print(action_pred)
                action = np.rint(action_pred)
                print("ACTION IS: {0}".format(action))
                obs1, r_t, terminal, info = env.step(action, actor_num, current_episode, x_t)
                print("Translation is: {0} and Rotation is: {1}".format(obs1[0:3], obs1[6:9]))

                total_rewards += r_t
                episode_length += 1
                total_steps += 1
                states_buf.append(obs1)
                rewards_buf.append(r_t)

                summary_writer.add_scalar('PPO_Positions/Final X Position', float(states_buf[-1][0]), episode_length)
                summary_writer.add_scalar('PPO_Positions/Final Y Position', float(states_buf[-1][1]), episode_length)
                summary_writer.add_scalar('PPO_Positions/Final Z Position', float(states_buf[-1][2]), episode_length)
                summary_writer.add_scalar('PPO_Rotations/Final Yaw Position', float(states_buf[-1][6]), episode_length)
                summary_writer.add_scalar('PPO_Rotations/Final Pitch Position', float(states_buf[-1][7]), episode_length)
                summary_writer.add_scalar('PPO_Rotations/Final Roll Position', float(states_buf[-1][8]), episode_length)

                if total_steps >=10:
                    if obs1[0] <= 0.2 and obs1[1] <= 0.2 and obs1[2] <= 0.2 and obs1[6] <= 0.2 and obs1[7] <= 0.2 and obs1[8] <= 0.2:
                        terminal = True

                
                obs = obs1

                if terminal:
                    global_episodes = global_episodes.assign_add(1)
                    print("global episode : {0}, episode length : {1}, , total reward : {2}".format(str(current_episode), str(episode_length), str(total_rewards)))

                    obs = env.reset()

                    summary_writer.add_scalar('PPO_Rewards/Mean_Rewards', float(sum(rewards_buf)/len(rewards_buf)), current_episode)
                    summary_writer.add_scalar('PPO_Rewards/Episode_Length', float(episode_length), current_episode)
                    summary_writer.add_scalar('PPO_Rewards/Episode_Reward', float(sum(rewards_buf)), current_episode)                      
                    summary_writer.add_scalar('PPO_Positions/Final X Position', float(states_buf[-1][0]), current_episode)
                    summary_writer.add_scalar('PPO_Positions/Final Y Position', float(states_buf[-1][1]), current_episode)
                    summary_writer.add_scalar('PPO_Positions/Final Z Position', float(states_buf[-1][2]), current_episode)
                    summary_writer.add_scalar('PPO_Rotations/Final Yaw Position', float(states_buf[-1][6]), current_episode)
                    summary_writer.add_scalar('PPO_Rotations/Final Pitch Position', float(states_buf[-1][7]), current_episode)
                    summary_writer.add_scalar('PPO_Rotations/Final Roll Position', float(states_buf[-1][8]), current_episode)
                    
                    total_rewards = 0
                    episode_length = 0

                    current_episode += 1
                    print("Episode Number: {0}".format(current_episode))
                    break 

                summary_writer.flush()

#Start testing:

print("STARTING TESTING SCRIPT")

tf.compat.v1.reset_default_graph()

env = gym.make("SpaceX_PPO_PROLONETS-v0")
global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
summary_writer = SummaryWriter(Config.TRAINING_LOG_PATH)
model_path = Config.CONTRON_PATH
print(model_path)

actor = build_net()
load_model(actor, model_path)

process()
