"""
Author: Steve Paul 
Date: 6/18/22 """
import pickle
from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv
from MRTA_Flood_Env import MRTA_Flood_Env
from training_config import get_config
import os

config = get_config()
n_test_Scenarios = 10 # number of test scenarios
env_list = []
for i in range(n_test_Scenarios):
    env = DummyVecEnv([lambda: MRTA_Flood_Env(
            n_locations = config.n_locations,
            n_agents = config.n_robots,
            max_capacity = config.max_capacity,
            max_range = config.max_range,
            enable_dynamic_tasks=config.enable_dynamic_tasks,
            display = False,
            enable_topological_features = config.enable_topological_features
    )])
    env_list.append(env)

if config.enable_dynamic_tasks:
    task_type = "D"
else:
    task_type = "ND"
mode =  0o755
path =  "Test_data/" + config.problem + "/"
if not os.path.exists(path):
    os.makedirs(path, mode)
file_name = path + config.problem\
                        + "_nloc_" + str(config.n_locations)\
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + ".pkl"
with open(file_name, 'wb') as fl:
    pickle.dump(env_list, fl)
fl.close()