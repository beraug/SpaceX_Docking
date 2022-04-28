#!/usr/bin/env python

class Config:
    NUM_EPISODE = 100
    NUM_STEPS = 2000
    NUM_BATCH = 3500

    LEARNING_RATE_A1 = 1e-6
    LEARNING_RATE_A2 = 1e-5
    LEARNING_RATE_A3 = 1e-5
    LEARNING_RATE_C = 1e-6
    CLIP_RANGE = 0.2
    ENTROPY_LOSS = 5e-3
    GAMMA = 0.95
    EPOCHS = 12
    STEPS_PER_EPOCH = 50
    BATCH_SIZE = 50

    HIDDEN_SIZE = 128
    NUM_LAYERS = 1

    ENV_NAME = 'SpaceX_ISS_docking'
    BETA1 = 0.99
    BETA2 = 0.90 

    #For random actions - not in use
    EPSILON_DECAY = 0.9999
    EPSILON_FLOOR = 0.0001
    EPSILON = 0.5

    TRAINING_LOG_PATH = r"C:/opt/SpaceX_Docking/Network_results_V00/"
    CONTRON_PATH = r"C:/opt/SpaceX_Docking/Contron_actor/"

  
