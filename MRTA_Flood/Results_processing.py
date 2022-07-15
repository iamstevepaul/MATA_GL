"""
Author: Steve Paul 
Date: 7/4/22 """

import warnings
from stable_baselines3 import PPO
from MRTA_Flood_Env import MRTA_Flood_Env
import torch
from topology import *
import pickle
import os
from CustomPolicies import ActorCriticGCAPSPolicy
from training_config import get_config
from stable_baselines3.common.vec_env import DummyVecEnv #, SubprocVecEnv
import gc
gc.collect()
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

config = get_config()
test = True  # if this is set as true, then make sure the test data is generated.
# Otherwise, run the test_env_generator script
if config.enable_dynamic_tasks:
    task_type = "D"
else:
    task_type = "ND"
enc_list = ["AM", "Feas_RND"]

config.device = torch.device("cuda:0" if config.use_cuda else "cpu")
if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
    tb_logger_location = config.logger+config.problem\
                     + "/" + config.node_encoder + "/" \
                    + config.problem\
                          + "_nloc_" + str(config.n_locations)\
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                         + config.node_encoder\
                         + "_K_" + str(config.K) \
                         + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
    save_model_loc = config.model_save+config.problem\
                     + "/" + config.node_encoder + "/" \
                    + config.problem\
                          + "_nloc_" + str(config.n_locations)\
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                         + config.node_encoder\
                         + "_K_" + str(config.K) \
                         + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
elif config.node_encoder == "AM":
    tb_logger_location = config.logger + config.problem \
                         + "/" + config.node_encoder + "/" \
                         + config.problem \
                         + "_nloc_" + str(config.n_locations) \
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_" \
                         + config.node_encoder \
                         + "_n_heads_" + str(config.n_heads) \
                         + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
    save_model_loc = config.model_save + config.problem \
                     + "/" + config.node_encoder + "/" \
                     + config.problem \
                     + "_nloc_" + str(config.n_locations) \
                     + "_nrob_" + str(config.n_robots) + "_" + task_type + "_" \
                     + config.node_encoder \
                     + "_n_heads_" + str(config.n_heads) \
                     + "_Le_" + str(config.Le) \
                     + "_h_" + str(config.features_dim)

if test:

    trained_model_n_loc = config.n_locations
    trained_model_n_robots = config.n_robots
    loc_test_multipliers = [0.5,1,2,5,10]
    robot_test_multipliers = [0.5,1]
    path =  "Test_data/" + config.problem + "/"
    for loc_mult in loc_test_multipliers:
        for rob_mult in robot_test_multipliers:
            n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots) + 1
            n_loc_test = int(trained_model_n_loc*loc_mult)

            for enc in enc_list:

                config.node_encoder = enc

                # file_name = path + config.problem \
                #             + "_nloc_" + str(n_loc_test) \
                #             + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"

                if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
                    encoder = config.node_encoder\
                                             + "_K_" + str(config.K) \
                                             + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                                             + "_h_" + str(config.features_dim)
                elif config.node_encoder == "AM":
                    encoder = config.node_encoder \
                             + "_n_heads_" + str(config.n_heads) \
                             + "_Le_" + str(config.Le) \
                             + "_h_" + str(config.features_dim)
                elif config.node_encoder == "BIGMRTA":
                    encoder = config.node_encoder
                elif config.node_encoder == "Feas_RND":
                    encoder = config.node_encoder

                result_path = "Results/" + config.problem + "/"

                result_file = result_path + config.problem + "_nloc_" + str(n_loc_test) \
                              + "_nrob_" + str(n_robots_test) + "_" + task_type + "_" + encoder

                with open(result_file, 'rb') as fl:
                    data = pickle.load(fl)
                fl.close()

                print(n_loc_test, n_robots_test, enc, data["total_rewards"].mean(), data["total_rewards"].std())