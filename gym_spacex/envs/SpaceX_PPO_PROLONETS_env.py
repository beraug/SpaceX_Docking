#!/usr/bin/env python
#Base
import time
import math
import sys
import numpy as np
import os

#GYM
import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

#Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

from Config_data_SpaceX import Config as Config

class SpaceXPpoProlonetsEnv(gym.Env):

    def __init__(self):

        options = Options()
        options.add_argument('--kiosk')

        self.driver =  webdriver.Firefox(executable_path=r'C:/opt/SpaceX_Docking/geckodriver-v0.30.0-win64/geckodriver.exe', options=options)
        self.driver.get('https://iss-sim.spacex.com/')
        WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'begin-button'))).click()
        WebDriverWait(self.driver, 30).until(EC.visibility_of_element_located((By.ID, 'hud-darken')))
        
        self.step_counter = 0
        self.time_old = None
        self.x_old = None
        self.y_old = None
        self.z_old = None
        self.yaw_old = None
        self.pitch_old = None
        self.roll_old = None

        self.Init_pose = [200, 12, 30]
        self.Init_rot = [-10, -20, 15]
        self.reward_counter_x = 0
        self.reward_counter_y = 0
        self.reward_counter_z = 0
        self.reward_counter_yaw = 0
        self.reward_counter_pitch = 0
        self.reward_counter_roll = 0

        x, y, z, vel_x, vel_y, vel_z = [], [], [], [], [], []
        yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = [], [], [], [], [], []

    # Resets the environment and retrieve initial observation
    def reset(self):

        print("RESETING!")

        reset_test = None

        while reset_test is None:
            try:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'option-restart'))).click()
                reset_test = True
                print("reset from running")

            except:
                time.sleep(15)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'fail-button'))).click()
                reset_test = True
                print("reset from fail")

            else:
                time.sleep(5)
                print("waiting")

        time.sleep(5)
        self.step_counter = 0
        self.nextEvent = 0
        self.Init_pose = [200, 12, 30]
        self.Init_rot = [-10, -20, 15]
        self.time_old = time.time()
        self.x_old = None
        self.y_old = None
        self.z_old = None
        self.yaw_old = None
        self.pitch_old = None
        self.roll_old = None
        self.reward_counter_x = 0
        self.reward_counter_y = 0
        self.reward_counter_z = 0
        self.reward_counter_yaw = 0
        self.reward_counter_pitch = 0
        self.reward_counter_roll = 0
        print("Reset action")

        # take main observations
        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = self.take_observation()
        
        print("Initial time is: {0}".format(self.time_old))
        state = np.array([x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll])
        
        return state

    def step(self, action, actor_number, current_episode, observation):

        sleep_time = 0.5

        #Rotational Actor
        if actor_number == 1:
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'yaw-left-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'yaw-right-button'))).click()
                print("yaw_left - action 0")
            elif action == 1:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'yaw-right-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'yaw-left-button'))).click()
                print("yaw_right - action 1")
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()

        elif actor_number == 2:
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'pitch-up-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'pitch-down-button'))).click()
                print("pitch_up - action 0")                
            elif action == 1:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'pitch-down-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'pitch-up-button'))).click()
                print("pitch_down - action 1")
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()

        elif actor_number == 3:
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'roll-left-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'roll-right-button'))).click()
                print("roll_left - action 0")
            elif action == 1:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'roll-right-button'))).click()
                time.sleep(sleep_time)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'roll-left-button'))).click()
                print("roll_right - action 1")                
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-rotation'))).click()

        #Translation
        elif actor_number == 4:
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-right-button'))).click()
                time.sleep(sleep_time*2)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-left-button'))).click()
                print("right - action 0")
            elif action == 1:               
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-left-button'))).click()
                time.sleep(sleep_time*2)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-right-button'))).click()
                print("left - action 1")
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()
                
        elif actor_number == 5:  
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-up-button'))).click()
                time.sleep(sleep_time*2)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-down-button'))).click()
                print("up - action 0")
            elif action == 1:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-down-button'))).click()
                time.sleep(sleep_time*2)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-up-button'))).click()
                print("down - action 1")
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()

        elif actor_number == 6:
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()
            if action == 0:
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-backward-button'))).click()
                time.sleep(sleep_time*2)
                WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-forward-button'))).click()
                print("backward - action 0")
            elif action == 1:
                if observation <= 2:
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-forward-button'))).click()
                    time.sleep(sleep_time*2)
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-backward-button'))).click()
                    print("forward - action 1")
                else:
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-forward-button'))).click()
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-forward-button'))).click()
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-forward-button'))).click()
                    time.sleep(sleep_time*2)
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-backward-button'))).click()
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-backward-button'))).click()
                    WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'translate-backward-button'))).click()
                    print("forward - action 1")
            elif action == 2:
                time.sleep(sleep_time*2)
                print("do nothing - action 2")
            WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.ID, 'toggle-translation'))).click()
            
        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = self.take_observation()

        self.step_counter += 1
        print("Step Counter = " + str(self.step_counter))

        training_log_path = Config.TRAINING_LOG_PATH
        screenshot_path = training_log_path + "Screenshots/" + "Episode_" + str(current_episode) + "/"
        if not os.path.exists(screenshot_path):
            os.makedirs(screenshot_path)
        if self.step_counter % 5 == 0:
            print("SCREENSHOT TAKEN")
            screenshot_path2 = screenshot_path + "Episode_" + str(current_episode) + "_Step_00" + str(self.step_counter) + ".png"
            print(screenshot_path2)
            self.driver.save_screenshot(screenshot_path2)

        # process data to check status
        reward, done = self.process_data(actor_number)

        if action == 0 or action == 1:
            reward += 0.0001

        if self.step_counter == 1200:
            done = True
            print("Step limit reached!")

        state = np.array([x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll])
        return state, reward, done, {}

    def take_observation(self):

        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = None, None, None, None, None, None, None, None, None, None, None, None
        while x==None or y==None or z==None or vel_x==None or vel_y==None or vel_z==None or yaw==None or pitch==None or roll==None or vel_yaw==None or vel_pitch==None or vel_roll==None:

                try:
                    x1 = self.driver.find_element_by_id('x-range').find_element_by_class_name('distance')
                    x = float(x1.text[:-2])
                    y1 = self.driver.find_element_by_id('y-range').find_element_by_class_name('distance')
                    y = float(y1.text[:-2])
                    z1 = self.driver.find_element_by_id('z-range').find_element_by_class_name('distance')
                    z = float(z1.text[:-2])
                    pos1 = self.driver.find_element_by_id('range').find_element_by_class_name('rate') 
                    pos = float(pos1.text[:-4])
                    yaw1 = self.driver.find_element_by_id('yaw').find_element_by_class_name('error')
                    yaw = float(yaw1.text[:-1])
                    pitch1 = self.driver.find_element_by_id('pitch').find_element_by_class_name('error')
                    pitch = float(pitch1.text[:-1])
                    roll1 = self.driver.find_element_by_id('roll').find_element_by_class_name('error')
                    roll = float(roll1.text[:-1])
                    vel_yaw1 = self.driver.find_element_by_id('yaw').find_element_by_class_name('rate')
                    vel_yaw = float(vel_yaw1.text[:-4])
                    vel_pitch1 = self.driver.find_element_by_id('pitch').find_element_by_class_name('rate')
                    vel_pitch = float(vel_pitch1.text[:-4])
                    vel_roll1 = self.driver.find_element_by_id('roll').find_element_by_class_name('rate')
                    vel_roll = float(vel_roll1.text[:-4])
                    vel_pos1 = self.driver.find_element_by_id('rate').find_element_by_class_name('rate')
                    vel_pos = float(vel_pos1.text[:-4])

                    vel_x, vel_y, vel_z = self.vel_calcs(x, y, z, pos, vel_pos)

                    #print("ALL POSITIONS IN: {0}/{1}/{2}/{3}/{4}/{5}".format(x, y, z, yaw, pitch, roll))
                    #print("ALL VELOCITIES IN : {0}/{1}/{2}/{3}/{4}/{5}".format(vel_x, vel_y, vel_z, vel_yaw, vel_pitch, vel_roll))

                    return x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll

                except:
                    print("not ready for messages")
                    self.reset()


    def vel_calcs(self, x, y, z, pos, vel_pos):       
        
        sin_x = x/pos
        sin_y = y/pos
        sin_z = z/pos

        vel_x = vel_pos*np.sin(sin_x)
        vel_y = vel_pos*np.sin(sin_y)
        vel_z = vel_pos*np.sin(sin_z)

        return vel_x, vel_y, vel_z
        

    def process_data(self, actor_num):

        done = False
        x, y, z, vel_x, vel_y, vel_z, yaw, pitch, roll, vel_yaw, vel_pitch, vel_roll = self.take_observation()
        eventPose = [x, y, z]
        PoseVel = [vel_x, vel_y, vel_z]
        #print("CURRENT POSITION IS: {0}".format(eventPose))
        targetPose = [0, 0, 0]
        targetRot = [0, 0, 0]
        eventRot = [yaw, pitch, roll]
        RotVel = [vel_yaw, vel_pitch, vel_roll]

        total_dist = self.calculate_dist_between_two_Points(self.Init_pose, targetPose)
        print("Total distance = {0}".format(total_dist))
        current_dist = self.calculate_dist_between_two_Points(eventPose, targetPose)
        print("Current distance = {0}".format(current_dist))


        if eventPose[0] >= 250 or eventPose[1] >= 60 or eventPose[2] >= 65:
            reward = -1
            print("TOO FAR - RESETTING")
            done = True

        elif eventPose[0] <= -60 or eventPose[1] <= -60 or eventPose[2] <= -60:
            reward = -1
            print("TOO FAR - RESETTING")
            done = True

        elif current_dist >= 250:
            reward = -1
            print("TOO FAR - RESETTING")
            done = True

        elif current_dist < -60:
            reward = -1
            print("Ops passed the point - RESETTING")
            done = True

        elif eventRot[0] >= 30.0 or eventRot[1] >= 30.0 or eventRot[2] >= 30.0:
            reward = -1
            print("Rotation Off - RESETTING")
            done = True

        elif eventRot[0] <= -30.0 or eventRot[1] <= -30.0 or eventRot[2] <= -30.0:
            reward = -1
            print("Rotation Off - RESETTING")
            done = True

        elif np.linalg.norm(RotVel[0]) >= 1.0 or np.linalg.norm(RotVel[1]) >= 1.0 or np.linalg.norm(RotVel[2]) >= 1.0:
            reward = -1
            print("Rotation too fast - RESETTING")
            done = True

        elif np.linalg.norm(PoseVel[0]) >= 0.1 or np.linalg.norm(PoseVel[1]) >= 0.1 or np.linalg.norm(PoseVel[2]) >= 0.1:
            reward = -1
            print("Translation too fast - RESETTING")
            done = True

        elif current_dist <= 0.2 and eventRot[0] <= 0.2 and eventRot[1] <= 0.2 and eventRot[2] <= 0.2:
            reward = 10000
            done = True
            print("WE MADE IT!")

        else:

            if actor_num == 1:
                if np.linalg.norm(eventRot[0]) < np.linalg.norm(self.Init_rot[0]):
                    reward = 1 / (1 + np.square(eventRot[0]))
                    #print("Yaw better!: {0}".format(str(reward)))
                    self.Init_rot[0] = eventRot[0]
                else:
                    reward = 0
                    #print("Yaw worst!: {0}".format(str(reward)))

                if np.linalg.norm(eventRot[1]) < np.linalg.norm(self.Init_rot[1]):
                    reward += 1 / (1 + np.square(eventRot[1]))
                    #print("Yaw better!: {0}".format(str(reward)))
                    self.Init_rot[1] = eventRot[1]
                else:
                    reward += 0
                    #print("Yaw worst!: {0}".format(str(reward)))

                if np.linalg.norm(eventRot[2]) < np.linalg.norm(self.Init_rot[2]):
                    reward += 1 / (1 + np.square(eventRot[2]))
                    #print("Yaw better!: {0}".format(str(reward)))
                    self.Init_rot[2] = eventRot[2]
                else:
                    reward += 0
                    #print("Yaw worst!: {0}".format(str(reward)))

            else:
                if np.linalg.norm(eventPose[0]) < np.linalg.norm(self.Init_pose[0]):
                    reward = 1 / (1 + np.square(current_dist))
                    #print("Getting closer on X!: {0}".format(str(reward)))
                    self.Init_pose[0] = eventPose[0]
                else:
                    reward = 0
                    #print("Moving away on X!: {0}".format(str(reward)))

                if np.linalg.norm(eventPose[1]) < np.linalg.norm(self.Init_pose[1]):
                    reward += 1 / (1 + np.square(current_dist))
                    #print("Getting closer on Y!: {0}".format(str(reward)))
                    self.Init_pose[1] = eventPose[1]
                else:
                    reward += 0
                    #print("Moving away on Y!: {0}".format(str(reward)))

                if np.linalg.norm(eventPose[2]) < np.linalg.norm(self.Init_pose[2]):
                    reward += 1 / (1 + np.square(current_dist))
                    #print("Getting closer on Z!: {0}".format(str(reward)))
                    self.Init_pose[1] = eventPose[1]
                else:
                    reward += 0
                    #print("Moving away on Z!: {0}".format(str(reward)))

                if current_dist < total_dist:
                    reward += 1 / (1 + np.square(current_dist))
                    #print("Getting closer!: {0}".format(str(reward)))
                else:
                    reward += 0
                    #print("Moving away!: {0}".format(str(reward)))

        return reward, done

    #Distance calculations
    def calculate_dist_between_two_Points(self, p_init, p_end):
        a = np.array((p_init[0], p_init[1], p_init[2]))
        b = np.array((p_end[0], p_end[1], p_end[2]))

        dist = np.linalg.norm(a - b)

        return dist
