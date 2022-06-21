"""
Author: Steve Paul 
Date: 4/15/22 """
""
import warnings
from stable_baselines3 import PPO
from MRTA_Flood_Env import MRTA_Flood_Env
import torch
from topology import *
import pickle
import os
from CustomPolicies import ActorCriticGCAPSPolicy
from training_config import get_config
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

def as_tensor(observation):
    for key, obs in observation.items():
        observation[key] = torch.tensor(obs)
    return observation

config = get_config()
test = True  # if this is set as true, then make sure the test data is generated.
# Otherwise, run the test_env_generator script
config.device = torch.device("cuda:0" if config.use_cuda else "cpu")
env = DummyVecEnv([lambda: MRTA_Flood_Env(
        n_locations = config.n_locations,
        n_agents = config.n_robots,
        max_capacity = config.max_capacity,
        max_range = config.max_range,
        enable_dynamic_tasks=config.enable_dynamic_tasks,
        display = False,
        enable_topological_features = config.enable_topological_features
)])

policy_kwargs=dict(
    # features_extractor_class=GCAPCNFeatureExtractor,
    features_extractor_kwargs=dict(
        feature_extractor=config.node_encoder,
        features_dim=config.features_dim,
        K=config.K,
        Le=config.Le,
        P=config.P,
        node_dim=env.envs[0].task_graph_node_dim,
        agent_node_dim=env.envs[0].agent_node_dim,
        n_heads=config.n_heads,
        tanh_clipping=config.tanh_clipping,
        mask_logits=config.mask_logits,
        temp=config.temp
    ),
    device=config.device
)

if config.enable_dynamic_tasks:
    task_type = "D"
else:
    task_type = "ND"


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

model = PPO(
    ActorCriticGCAPSPolicy,
    env,
    gamma=config.gamma,
    verbose=1,
    n_epochs=config.n_epochs,
    batch_size=config.batch_size,
    tensorboard_log=tb_logger_location,
    # create_eval_env=True,
    n_steps=config.n_steps,
    learning_rate=config.learning_rate,
    policy_kwargs = policy_kwargs,
    ent_coef=config.ent_coef,
    vf_coef=config.val_coef,
    device=config.device
)
model.learn(total_timesteps=config.total_steps)

obs = env.reset()
model.save(save_model_loc)
if test:
    model = PPO.load(save_model_loc, env=env)

    trained_model_n_loc = config.n_locations
    trained_model_n_robots = config.n_robots
    loc_test_multipliers = [0.5,1,2,5,10]
    robot_test_multipliers = [0.5,1,2]
    path =  "Test_data/" + config.problem + "/"
    for loc_mult in loc_test_multipliers:
        for rob_mult in robot_test_multipliers:
            n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots)
            n_loc_test = int(trained_model_n_loc*loc_mult)

            env = DummyVecEnv([lambda: MRTA_Flood_Env(
                    n_locations = n_loc_test,
                    n_agents = n_robots_test,
                    max_capacity = config.max_capacity,
                    max_range = config.max_range,
                    enable_dynamic_tasks=config.enable_dynamic_tasks,
                    display = False,
                    enable_topological_features = config.enable_topological_features
            )])

            file_name = path + config.problem\
                                    + "_nloc_" + str(n_loc_test)\
                                     + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
            with open(file_name, 'rb') as fl:
                test_envs = pickle.load(fl)
            fl.close()
            total_rewards_list = []
            distance_list = []
            total_tasks_done_list = []
            for env in test_envs:
                env.envs[0].training = False
                model.env = env
                obs = env.reset()
                obs = as_tensor(obs)
                for i in range(1000000):
                        model.policy.set_training_mode(False)
                        action = model.policy._predict(obs)
                        obs, reward, done, _ = env.step(action)
                        obs = as_tensor(obs)
                        if done:
                                total_rewards_list.append(reward)
                                distance_list.append(env.envs[0].total_distance_travelled)
                                total_tasks_done_list.append(env.envs[0].task_done.sum())
                                break

            total_rewards_array = np.array(total_rewards_list)
            distance_list_array = np.array(distance_list)
            total_tasks_done_array = np.array(total_tasks_done_list)
            if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
                encoder = config.node_encoder\
                                         + "_K_" + str(config.K) \
                                         + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                                         + "_h_" + str(config.features_dim)
            else:
                encoder = config.node_encoder \
                         + "_n_heads_" + str(config.n_heads) \
                         + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
            data = {
                "problem": config.problem,
                "n_locations": n_loc_test,
                "n_robots": n_robots_test,
                "dynamic_task": config.enable_dynamic_tasks,
                "policy":encoder,
                "total_tasks_done": total_tasks_done_array,
                "total_rewards": total_rewards_array,
                "distance": distance_list_array
            }

            result_path = "Results/" + config.problem + "/"

            result_file = result_path + config.problem + "_nloc_" + str(n_loc_test) \
                          + "_nrob_" + str(n_robots_test) + "_" + task_type + "_" + encoder
            mode = 0o755
            if not os.path.exists(result_path):
                os.makedirs(result_path, mode)
            with open(result_file, 'wb') as fl:
                pickle.dump(data, fl)
            fl.close()
