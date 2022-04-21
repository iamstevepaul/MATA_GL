"""
Author: Steve Paul 
Date: 4/15/22 """
""
from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from mrta_flood_env import MRTAENV
import json
import datetime as dt
import torch
from utils import *
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot
from CustomPolicies import ActorCriticGCAPSPolicy
# from CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines_al.common.utils import set_random_seed


from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv

warnings.filterwarnings('ignore')

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


# class CCN(nn.Module):
#
#     def __init__(
#             self,
#             n_layers=2,
#             features_dim=128,
#             n_p=1,
#             node_dim=2,
#             n_K=1,
#             n_neighbors = 3
#     ):
#         super(CCN, self).__init__()
#         self.init_embed = nn.Linear(node_dim, features_dim)
#         self.n_neighbors = n_neighbors
#         self.init_neighbour_embed = nn.Linear(node_dim, features_dim)
#         self.neighbour_encode = nn.Linear(features_dim, features_dim)
#         self.neighbour_encode_2 = nn.Linear(features_dim, features_dim)
#         self.init_embed_depot = nn.Linear(2, features_dim)
#         self.final_embedding = nn.Linear(features_dim, features_dim)
#         self.final_embedding_2 = nn.Linear(features_dim, features_dim)
#         self.features_dim=features_dim
#         self.activ = nn.LeakyReLU()
#         self.full_context_nn = torch.nn.Linear(4 * features_dim, features_dim)
#         self.agent_decision_context = torch.nn.Linear(2, features_dim)
#         self.agent_context = torch.nn.Linear(2, features_dim)
#         self.agent_mask_encoding = torch.nn.Linear(71, features_dim)
#
#     def forward(self, X_l, mask=None):
#         x = X_l['location'][:, 1:, :]
#         # X = torch.cat((data['loc'], data['deadline']), -1)
#         X_loc = X_l['location'][:, 1:, :]
#         dist_mat = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
#         # x2 = x[:, :, 0:3]
#
#         # F0_embedding_2d = self.init_embed_2d(x2)
#         F0_embedding_3d = self.init_embed(x)
#         # F0_embedding.reshape([1])
#
#         # dist_mat = (x[:, None] - x[:, :, None]).norm(dim=-1, p=2)  ## device to cuda to be added
#         neighbors = dist_mat.sort().indices[:, :, :self.n_neighbors]  # for 6 neighbours
#         # neighbour = x[:, neighbors][0]
#         # neighbour_delta = neighbour - x[:, :, None, :]
#         neighbour_delta_embedding = self.neighbour_encode(F0_embedding_3d[:, neighbors][0] - F0_embedding_3d[:, :, None, :])
#         concat = torch.cat((F0_embedding_3d[:, :, None, :], neighbour_delta_embedding), 2)
#         F_embed_final = self.final_embedding(concat).sum(2)
#
#         # neighbour_delta_embedding_2 = self.activ(
#         #     self.neighbour_encode_2(F_embed_final[:, neighbors][0] - F_embed_final[:, :, None, :]))
#         # concat_2 = torch.cat((F_embed_final[:, :, None, :], neighbour_delta_embedding_2), 2)
#         # F_embed_final_2 = self.final_embedding_2(concat_2).sum(2)
#         #h2_neighbor = F_embed_final[:, neighbors][0]
#         #F_embed_final_2 = self.neighbour_encode_2(h2_neighbor).sum(dim=2)
#         init_depot_embed = self.init_embed_depot(X_l['depot'])
#         h = torch.cat((init_depot_embed, F_embed_final), -2)
#         context = self.full_context_nn(
#             torch.cat(
#                 (h.mean(dim=1)[:, None, :], self.agent_decision_context(X_l['agent_taking_decision_coordinates']),
#                  self.agent_context(X_l['agents_destination_coordinates']).sum(1)[:, None, :],
#                  self.agent_mask_encoding(X_l['mask'].permute(0, 2, 1))),
#                 -1))
#         mask_shape = X_l['mask'].shape
#         return context
#



#
# class GCAPCNFeatureExtractor2(nn.Module):
#
#     def __init__(self,
#                  n_layers=2,
#                  features_dim=128,
#                  n_p=1,
#                  node_dim=2,
#                  n_K=1
#                  ):
#         super(GCAPCNFeatureExtractor2, self).__init__()
#         self.n_layers = n_layers
#         self.n_dim = features_dim
#         self.features_dim=features_dim
#         self.n_p = n_p
#         self.n_K = n_K
#         self.node_dim = node_dim
#         self.init_embed = nn.Linear(node_dim, features_dim * n_p)
#         self.init_embed_depot = nn.Linear(2, features_dim)
#
#         self.W_L_1_G1 = nn.Linear(features_dim * (n_K + 1) * n_p, features_dim)
#
#         self.normalization_1 = nn.BatchNorm1d(features_dim * n_p)
#
#         self.W_F = nn.Linear(features_dim * n_p, features_dim)
#         self.full_context_nn = torch.nn.Linear(11+10+3+2, features_dim)
#         self.agent_decision_context = torch.nn.Linear(2, features_dim)
#         self.agent_context = torch.nn.Linear(2, features_dim)
#         self.agent_mask_encoding = torch.nn.Linear(11, features_dim)
#
#         self.activ = nn.Tanh()
#
#     def forward(self, data, mask=None):
#
#         X = data['location'][:,1:,:]
#         # distance_matrix = ((((X[:, :, None] - X[:, None]) ** 2).sum(-1)) ** .5)[0]
#
#
#
#         num_samples, num_locations, _ = X.size()
#         # A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
#         #     (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
#         # A[A != A] = 0
#         # D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
#         #               (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
#
#         # Layer 1
#
#         # p = 3
#         F0 = self.init_embed(X)
#
#         # K = 3
#         # L = D - A
#         L_topo = data["topo_laplacian"]
#         L = L_topo
#         # L_squared = torch.matmul(L, L)
#         # L_cube = torch.matmul(L, L_squared)
#
#         g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
#                                           torch.matmul(L, F0)[:, :, :]
#                                           # torch.matmul(L_squared, F0)[:, :, :]
#                                           ),
#                                          -1))
#
#
#         F1 = g_L1_1#torch.cat((g_L1_1), -1)
#         F1 = self.activ(F1) #+ F0
#         # F1 = self.normalization_1(F1)
#
#         F_final = self.activ(self.W_F(F1))
#
#         # init_depot_embed = self.init_embed_depot(data['depot'])[:]
#         h = F_final#torch.cat((init_depot_embed, F_final), 1)
#
#         context = self.full_context_nn(
#                       torch.cat((h.mean(dim=2)[:, None, :], data['agent_taking_decision_coordinates'],
#                               self.agent_context(data['agents_destination_coordinates']).sum(2)[:, None, :],
#                                 data['mask'].permute(0,2,1)),
#                              -1))
#         # mask_shape = data['mask'].shape
#         return self.activ(context)

env = DummyVecEnv([lambda: MRTAENV(
        n_locations = 31,
        n_agents = 3
)])

# n_envs = 4
# env = make_vec_env(mTSPEnv, n_envs=n_envs, env_kwargs={"n_locations":21, "n_agents":5})
# num_cpu = 6
# env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

# model = PPO('MlpPolicy',
#     env,
#     gamma=1.00,
#     verbose=1,
#     n_epochs=10,
#     batch_size=10000,
#     tensorboard_log="logger/",
#     create_eval_env=True,
#     n_steps=20000)

policy_kwargs=dict(
    # features_extractor_class=GCAPCNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128,node_dim=2),
    # activation_fn=torch.nn.LeakyReLU,
    # net_arch=[dict(vf=[128,128])]
)

model = PPO(
    ActorCriticGCAPSPolicy,
    env,
    gamma=1.00,
    verbose=1,
    n_epochs=100,
    batch_size=10000,
    tensorboard_log="logger/",
    # create_eval_env=True,
    n_steps=20000,
    learning_rate=0.000001,
    policy_kwargs = policy_kwargs,
    ent_coef=0.01,
    vf_coef=0.5
)
#
# model = A2C(
#     ActorCriticGCAPSPolicy,
#     env,
#     gamma=1.00,
#     verbose=1,
#     tensorboard_log="logger/",
#     create_eval_env=True,
#     n_steps=20000
# )

model.learn(total_timesteps=2000000)

obs = env.reset()

log_dir = "."
model.save(log_dir + "r1")
model = PPO.load(log_dir + "r1", env=env)