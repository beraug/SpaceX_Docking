#!/usr/bin/env python

#Base
import os
import random
import numpy as np
import datetime

#Tensorflow
import tensorflow as tf 
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
import numba as nb
from tensorboardX import SummaryWriter

#Other scripts
import SpaceX_PPO_PROLONETS as PPO_model
from Config_data_SpaceX import Config as Config


class Worker:

    def __init__(self, name, env, summary_writer, global_episodes):
        self.env = env
        self.summary_writer = summary_writer
        self.global_episodes = global_episodes
        self.increase_global_episodes = self.global_episodes.assign_add(1)
        self.gamma = Config.GAMMA
        self.epochs = Config.EPOCHS
        #self.steps_per_epoch = Config.STEPS_PER_EPOCH
        self.steps_per_epoch = 1
        self.num_steps = Config.NUM_STEPS
        self.batch_count = 0
        self.batch_size = Config.BATCH_SIZE

        self.turn_LSTM = True

        self.lr_A1 = Config.LEARNING_RATE_A1
        self.lr_A2 = Config.LEARNING_RATE_A2
        self.lr_A3 = Config.LEARNING_RATE_A3
        
        self.PPO = PPO_model.PPO()

        self.checkpoint_path = self.PPO.checkpoint_path

        self.num_state = self.PPO.num_state
        self.num_action = self.PPO.num_action
        self.dummy_action = np.zeros((1, self.num_action), dtype=np.float32)
        self.dummy_value = np.zeros((1, 1), dtype=np.float32)
        self.dummy_state = np.zeros((1, self.num_state), dtype=np.float32)

        if self.turn_LSTM is True:
            self.actor1 = self.PPO.build_actor_lstm_net(self.lr_A1)
            self.critic1 = self.PPO.build_critic_lstm_net()
        
            self.actor2 = self.PPO.build_actor_lstm_net(self.lr_A2)
            self.critic2 = self.PPO.build_critic_lstm_net()

            self.actor3 = self.PPO.build_actor_lstm_net(self.lr_A3)
            self.critic3 = self.PPO.build_critic_lstm_net()

        else:
            self.actor1 = self.PPO.build_actor_net(self.lr_A1)
            self.critic1 = self.PPO.build_critic_net()
        
            self.actor2 = self.PPO.build_actor_net(self.lr_A2)
            self.critic2 = self.PPO.build_critic_net()

            self.actor3 = self.PPO.build_actor_net(self.lr_A3)
            self.critic3 = self.PPO.build_critic_net()

    def choose_action(self, observation, val, actor_num, reward, old_pred):

        print(reward)
        print(old_pred)
        adv = reward * self.gamma
        adv = np.array(adv)
        adv = adv.reshape(1)
        old_pred = np.array(old_pred)
        old_pred = old_pred.reshape(1, self.num_action)
        print(adv.shape)
        print(old_pred.shape)
        if actor_num == 1:
            p = self.actor1.predict([observation.reshape(1, self.num_state), adv, old_pred])
            print("ACTOR1 PREDICTION IS: {0}".format(p[0]))
        elif actor_num ==2:
            p = self.actor2.predict([observation.reshape(1, self.num_state), adv, old_pred])
            print("ACTOR2 PREDICTION IS: {0}".format(p[0]))
        else:
            p = self.actor3.predict([observation.reshape(1, self.num_state), adv, old_pred])
            print("ACTOR3 PREDICTION IS: {0}".format(p[0]))

        if val is False:
            action = np.random.choice(self.num_action, p=np.nan_to_num(p[0]))
            #action = np.argmax(p[0])
        else:
            action = np.argmax(p[0])
            #action = np.random.choice(self.num_action, p=np.nan_to_num(p[0]))

        action_matrix = np.zeros(self.num_action)
        action_matrix[action] = 1
        return action, action_matrix, p

    def pick_actor(self, obs, init_episodes):
        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = obs

        #if init_episodes <= -5:
        #    actor_num = 2
        #elif -5 < init_episodes <= -1:
        #    actor_num = 3
        #else:
        if np.linalg.norm(yaw) >= 0.2 or np.linalg.norm(pitch) >= 0.2 or np.linalg.norm(roll) >= 0.2:
            actor_num = 1
        elif np.linalg.norm(x) >= 2 or np.linalg.norm(y) >= 2 or np.linalg.norm(z) >= 2:
            actor_num = 2
        else:
            actor_num = 3

        return actor_num

    def process(self):

        obs = self.env.reset()
        x_t1 = []
        total_rewards = 0
        self.episode = 0
        episode_length = 0
        total_steps = 0
        self.val = False
        self.summaries = tf.compat.v1.summary.merge_all() 

        while True:
            states_buf = [[self.dummy_state]]
            actions_buf = [[self.dummy_action]]
            rewards_buf = [0]
            action_matrix_buf = [[0, 0, 0, 0, 0, 0, 1]]
            #action_matrix_buf = [[0, 0, 1]]
            pred_action_buf = [[0, 0, 0, 0, 0, 0, 1]]
            #pred_action_buf = [[0, 0, 1]]
            
            states_buf1 = []
            actions_buf1 = []
            rewards_buf1 = []
            action_matrix_buf1 = []
            pred_action_buf1 = []

            states_buf2 = []
            actions_buf2 = []
            rewards_buf2 = []
            action_matrix_buf2 = []
            pred_action_buf2 = []

            states_buf3 = []
            actions_buf3 = []
            rewards_buf3 = []
            action_matrix_buf3 = []
            pred_action_buf3 = []

            if self.episode % 6 == 0:
                self.val = True
                print("Evaluating the Network in the next episode")
            else:
                self.val = False

            for i in range(0, self.num_steps):

                actor_num = self.pick_actor(obs, self.episode)
                if actor_num == 1:
                    x_t = obs[6:12]
                    print("ROT OBS IS: {0}".format(x_t))
                else:
                    x_t = obs[:6]
                    print("TRANS OBS IS: {0}".format(x_t))

                action, action_matrix, predicted_action = self.choose_action(x_t, self.val, actor_num, rewards_buf[-1], pred_action_buf[-1])
                print("ACTION IS: {0}".format(action))
                print("ACTION MATRIX IS: {0}".format(action_matrix))
                print("Predicted ACTION IS: {0}".format(predicted_action))
                obs1, r_t, self.terminal, info = self.env.step(action, actor_num)

                total_rewards += r_t
                episode_length += 1
                total_steps += 1
                states_buf.append(obs1)
                actions_buf.append(action)
                action_matrix_buf.append(action_matrix)
                pred_action_buf.append(predicted_action)
                rewards_buf.append(r_t)
                    
                if actor_num == 1:
                    x_t1 = obs1[6:12]
                    states_buf1.append(x_t1)
                    actions_buf1.append(action)
                    action_matrix_buf1.append(action_matrix)
                    pred_action_buf1.append(predicted_action)
                    rewards_buf1.append(r_t)
                elif actor_num == 2:
                    x_t1 = obs1[:6]
                    states_buf2.append(x_t1)
                    actions_buf2.append(action)
                    action_matrix_buf2.append(action_matrix)
                    pred_action_buf2.append(predicted_action)
                    rewards_buf2.append(r_t)
                else:
                    x_t1 = obs1[:6]
                    states_buf3.append(x_t1)
                    actions_buf3.append(action)
                    action_matrix_buf3.append(action_matrix)
                    pred_action_buf3.append(predicted_action)
                    rewards_buf3.append(r_t)

                obs = obs1

                if self.terminal:
                    global_episodes = self.global_episodes.assign_add(1)
                    print('ID :' + ', global episode :' + str(self.episode) + ', episode length :' + str(episode_length) + ', total reward :' + str(total_rewards))
                    self.training_step = episode_length
                    obs = self.env.reset()
                    #self.summary_writer.add_scalar(tag='PPO_Rewards/Total_Rewards', simple_value=float(total_rewards), global_episodes)
                    self.summary_writer.add_scalar('PPO_Rewards/Mean_Rewards', float(sum(rewards_buf)/len(rewards_buf)), self.episode)
                    self.summary_writer.add_scalar('PPO_Rewards/Episode_Length', float(episode_length), self.episode)
                    self.summary_writer.add_scalar('PPO_Rewards/Episode_Reward', float(sum(rewards_buf)), self.episode)
                        
                    self.summary_writer.add_scalar('PPO_Positions/Final X Position', float(states_buf[-1][0]), self.episode)
                    self.summary_writer.add_scalar('PPO_Positions/Final Y Position', float(states_buf[-1][1]), self.episode)
                    self.summary_writer.add_scalar('PPO_Positions/Final Z Position', float(states_buf[-1][2]), self.episode)
                    self.summary_writer.add_scalar('PPO_Rotations/Final Yaw Position', float(states_buf[-1][6]), self.episode)
                    self.summary_writer.add_scalar('PPO_Rotations/Final Pitch Position', float(states_buf[-1][7]), self.episode)
                    self.summary_writer.add_scalar('PPO_Rotations/Final Roll Position', float(states_buf[-1][8]), self.episode)
                    
                    #self.summary_writer.add_summary(summary, global_episodes)
                    #self.summary_writer.flush()
                    #grad_episode_length = episode_length
                    total_rewards = 0
                    episode_length = 0
                    #grad_episode = self.episode

                    if self.val is True:
                        self.evaluation_summary_write(rewards_buf, self.training_step, states_buf, self.episode)
                    
                    self.episode += 1
                    print("Episode Number: {0}".format(self.episode))
                    break   

            self.steps_per_epoch = int(np.ceil(total_steps/self.batch_size))
            print(self.steps_per_epoch)

            if self.episode % 5 == 0:
                #if actor_num == 1:

                discounted_r = self.transform_reward(rewards_buf1)
                bs, ba, bp, br = np.array(states_buf1), np.array(actions_buf1), np.array(pred_action_buf1), np.reshape(np.array(discounted_r), (len(discounted_r), 1))
                ba = K.cast(ba, tf.float32)

                actor_entropy = -np.mean(bp * np.log(bp + 1e-10))
                print("ACTOR 1 Entropy is: {0}".format(actor_entropy))

                print("BP HERE: {0}".format(bp))
                bp= np.reshape(bp, (bp.shape[0], bp.shape[2]))
                old_pred = bp
                print("old_pred HERE: {0}".format(old_pred))
                pred_values = self.critic1.predict(bs)

                advantage = br - pred_values

                actor_loss = self.actor1.fit([bs, advantage, old_pred], [ba], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)
                critic_loss = self.critic1.fit([bs], [br], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)
                        
                self.summary_writer.add_scalar('Actor_Losses/Actor_1_Loss', actor_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Critic_Losses/Critic_1_Loss', critic_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Actor_Entropy/Actor_1_Entropy',actor_entropy, self.episode)

                #self.PPO.save_model(self.actor1, self.episode)
                #self.PPO.save_model(self.critic1, self.episode)

                #elif actor_num == 2:    

                discounted_r = self.transform_reward(rewards_buf2)
                bs, ba, bp, br = np.array(states_buf2), np.array(actions_buf2), np.array(pred_action_buf2), np.reshape(np.array(discounted_r), (len(discounted_r), 1))
                ba = K.cast(ba, tf.float32)

                actor_entropy = -np.mean(bp * np.log(bp + 1e-10))
                print("ACTOR 2 Entropy is: {0}".format(actor_entropy))

                bp= np.reshape(bp, (bp.shape[0], bp.shape[2]))
                old_pred = bp
                pred_values = self.critic2.predict(bs)

                advantage = br - pred_values

                actor_loss = self.actor2.fit([bs, advantage, old_pred], [ba], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)
                critic_loss = self.critic2.fit([bs], [br], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)

                self.summary_writer.add_scalar('Actor_Losses/Actor_2_Loss', actor_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Critic_Losses/Critic_2_Loss', critic_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Actor_Entropy/Actor_2_Entropy', actor_entropy, self.episode)

                #self.PPO.save_model(self.actor2, self.episode)
                #self.PPO.save_model(self.critic2, self.episode)

                #else:

                discounted_r = self.transform_reward(rewards_buf3)
                bs, ba, bp, br = np.array(states_buf3), np.array(actions_buf3), np.array(pred_action_buf3), np.reshape(np.array(discounted_r), (len(discounted_r), 1))
                ba = K.cast(ba, tf.float32)

                actor_entropy = -np.mean(bp * np.log(bp + 1e-10))
                print("ACTOR 3 Entropy is: {0}".format(actor_entropy))

                bp= np.reshape(bp, (bp.shape[0], bp.shape[2]))
                old_pred = bp
                pred_values = self.critic3.predict(bs)

                advantage = br - pred_values

                actor_loss = self.actor3.fit([bs, advantage, old_pred], [ba], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)
                critic_loss = self.critic3.fit([bs], [br], steps_per_epoch=self.steps_per_epoch, batch_size=self.batch_size)

                self.summary_writer.add_scalar('Actor_Losses/Actor_3_Loss', actor_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Critic_Losses/Critic_3_Loss', critic_loss.history['loss'][-1], self.episode)
                self.summary_writer.add_scalar('Actor_Entropy/Actor_3_Entropy', actor_entropy, self.episode)

                #self.PPO.save_model(self.actor3, self.episode)
                #self.PPO.save_model(self.critic3, self.episode)

                self.PPO.save_model(self.actor1, self.episode)
                self.PPO.save_model(self.critic1, self.episode)

                self.PPO.save_model(self.actor2, self.episode)
                self.PPO.save_model(self.critic2, self.episode)

                self.PPO.save_model(self.actor3, self.episode)
                self.PPO.save_model(self.critic3, self.episode)

                self.summary_writer.flush()

    def transform_reward(self, rewards):
        
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * self.gamma

        return rewards

    def evaluation_summary_write(self, rewards_buf, training_step, states_buf, episode):

        self.summary_writer.add_scalar('Evaluation_Rewards/Episode_Reward', np.array(rewards_buf).sum(), episode)
        self.summary_writer.add_scalar('Evaluation_Rewards/Mean_Rewards', float(sum(rewards_buf)/len(rewards_buf)), episode)
        self.summary_writer.add_scalar('Evaluation_Rewards/Episode_Length', float(training_step), episode)
        self.summary_writer.add_scalar('Evaluation_Positions/Final X Position', float(states_buf[-1][0]), episode)
        self.summary_writer.add_scalar('Evaluation_Positions/Final Y Position', float(states_buf[-1][1]), episode)
        self.summary_writer.add_scalar('Evaluation_Positions/Final Z Position', float(states_buf[-1][2]), episode)
        self.summary_writer.add_scalar('Evaluation_Rotations/Final Yaw Position', float(states_buf[-1][6]), episode)
        self.summary_writer.add_scalar('Evaluation_Rotations/Final Pitch Position', float(states_buf[-1][7]), episode)
        self.summary_writer.add_scalar('Evaluation_Rotations/Final Roll Position', float(states_buf[-1][8]), episode)


        
