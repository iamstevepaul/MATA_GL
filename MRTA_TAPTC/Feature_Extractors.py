"""
Author: Steve Paul 
Date: 1/18/22 """

from torch import nn
import torch

class MLP(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 node_dim=2,
                 inter_dim=128
                 ):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.features_dim = features_dim
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, inter_dim)
        self.layer_1 = nn.Linear(inter_dim, inter_dim)
        self.WF = nn.Linear(inter_dim, features_dim)
        self.init_embed_depot = nn.Linear(2, features_dim)
        self.activ = nn.Tanh()


    def forward(self, data, mask=None):

        X = data['task_graph_nodes']
        F0 = self.init_embed(X)
        F0 = self.activ(self.self.layer_1(F0))
        F0 = self.activ(self.WF(F0))
        h = F0#torch.cat((init_depot_embed, F0), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class CAPAM(nn.Module):

    def __init__(self,
                 Le=2,
                 features_dim=128,
                 P=1,
                 node_dim=2,
                 K=1
                 ):
        super(CAPAM, self).__init__()
        self.Le = Le
        self.features_dim = features_dim
        self.P = P
        self.K = K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * P)
        self.init_embed_depot = nn.Linear(2, features_dim)

        self.W_L_1_G1 = nn.Linear(features_dim * (K + 1) * P, features_dim)

        self.W_F = nn.Linear(features_dim * P, features_dim)

        self.activ = nn.Tanh()

    def forward(self, data, mask=None):
        # active_tasks = ((data['nodes_visited'] == 0).nonzero())[:, 1]
        # print("Active tasks before node embedding: ",active_tasks)
        X = data['task_graph_nodes']

        num_samples, num_locations, _ = X.size()

        A = data['task_graph_adjacency']
        # A = data['task_graph_adjacency']
        D = torch.mul(torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))




        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        # F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])
        # K = 3
        L = D - A
        # L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          ),
                                         -1))

        # F1 = self.normalization_1(F1)
        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) #+ F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        # init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = F_final # torch.cat((init_depot_embed, F_final), 1)
        # print("Shape of the node embeddings: ", h.shape)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCNFeatureExtractor(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=1
                 ):
        super(GCAPCNFeatureExtractor, self).__init__()
        self.n_layers = n_layers
        self.n_dim = features_dim
        self.features_dim=features_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * n_p)
        self.init_embed_depot = nn.Linear(2, features_dim)

        self.W_L_1_G1 = nn.Linear(features_dim * (n_K + 1) * n_p, features_dim)

        self.normalization_1 = nn.BatchNorm1d(features_dim * n_p)

        self.W_F = nn.Linear(features_dim * n_p, features_dim)
        self.activ = nn.Tanh()

    def forward(self, data, mask=None):
        # active_tasks = ((data['nodes_visited'] == 0).nonzero())[:,1]

        X = data['task_graph_nodes']
        # X = X[:,active_tasks[1:]-1,:]
        # distance_matrix = ((((X[:, :, None] - X[:, None]) ** 2).sum(-1)) ** .5)[0]



        num_samples, num_locations, _ = X.size()

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3
        # L = D - A
        L_topo = data["topo_laplacian"]
        L = L_topo
        # L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          # torch.matmul(L_squared, F0)[:, :, :]
                                          ),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) #+ F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = torch.cat((init_depot_embed, F_final), 1)
        # print("Shape of the node embeddings: ", h.shape)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )