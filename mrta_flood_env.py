"""
Author: Steve Paul 
Date: 4/14/22 """
""
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import gym
from gym import Env
from collections import defaultdict
from gym.spaces import Discrete, MultiBinary, Box, Dict
import random
import torch
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot


# TODO:
#   Generate task graph -  done
#   Generate Feasible task graph - done
#   Generate agent network graph - done
#   parallelize PD computation - done
#   Trial runs for debugging
#   Include agent taking decision to the state dictionary - done
#   Introduce dynamic tasks:
#       Generation of dynamic tasks
#       Dynamic task graph generation
#   Introduce communication uncertainties
#   BiGraph matching based decoding
#   Move TD abstraction to policy file
#   Fix the memory issue


# Create Environment
class MRTAENV(Env):

    def __init__(self,
                 n_locations=100,
                 visited=[],
                 n_agents=2,
                 agents=[],
                 agents_location=[],
                 total_distance_travelled=0.0,
                 max_capacity=6,
                 max_range=4,
                 ):
        # Action will be choosing the next task. (Can be a task that is alraedy done)
        # It would be great if we can force the agent to choose not-done task
        super(MRTAENV, self).__init__()
        self.n_locations = n_locations
        self.action_space = Discrete(1)
        self.locations = np.random.random((n_locations, 2))
        self.depot = self.locations[0, :]
        self.visited = visited
        self.n_agents = n_agents
        self.agents = agents
        self.agents_location = agents_location
        self.agents_prev_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_distance_travelled = np.zeros((n_agents, 1))
        self.total_distance_travelled = total_distance_travelled
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((n_locations, 1))
        self.n_locations = n_locations
        self.agents_distance_to_destination = np.zeros(
            (n_agents, 1))  # tracks the distance to destination from its current point for all robots

        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agent_speed = 0.01
        self.agents_next_decision_time = np.zeros((n_agents, 1))
        self.agents_destination_coordinates = np.ones((n_agents, 1)) * self.depot

        self.state = 00  # call the graph encoding function + context here
        self.observation = 00  # call the graph encoding function + context here

        self.total_reward = 0.0
        self.total_length = 0
        self.first_dec = True

        self.max_capacity = max_capacity
        self.max_range = max_range
        self.agents_current_range = torch.ones((1,n_agents), dtype=torch.float32)*max_range
        self.agents_current_payload = torch.ones((1,n_agents), dtype=torch.float32)*max_capacity
        self.time_deadlines = (torch.tensor(np.random.random((1, n_locations)))*.3 + .7)*200
        self.time_deadlines[0, 0] = 1000000
        self.location_demand = torch.ones((1, n_locations), dtype=torch.float32)
        self.task_done = torch.zeros((1, n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, n_locations), dtype=torch.float32)
        self.depot_id = 0
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]


        self.observation_space = Dict(
            dict(
                # location=Box(low=0, high=1, shape=self.locations.shape),
                depot=Box(low=0, high=1, shape=(1, 2)),
                mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                # agents_destination_coordinates=Box(low=0, high=1, shape=self.agents_destination_coordinates.shape),
                # agent_taking_decision_coordinates=Box(low=0, high=1, shape=self.agents_destination_coordinates[
                #                                                            self.agent_taking_decision, :].reshape(1,
                #                                                                                                   2).shape),
                # topo_laplacian=Box(low=0, high=100000, shape=(n_locations-1,n_locations-1)),
                task_graph_nodes=Box(low=0, high=1, shape=(n_locations-1,4)),
                # task_graph_adjacency=Box(low=0, high=1, shape=(n_locations-1, n_locations-1)),
                agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, 5)),
                # agents_graph_adjacency=Box(low=0, high=1, shape=(n_agents, n_agents)),
                nodes_visited=Box(low=0, high=1, shape=self.nodes_visited.shape),
                agent_taking_decision=Discrete(n_agents),
                first_dec = MultiBinary(1)
            ))


        self.distance = 0.0
        self.topo_laplacian = None
        # state = self.get_encoded_state()
        # topo_laplacian = self.get_topo_laplacian(state)
        # state["topo_laplacian"] = topo_laplacian
        # self.topo_laplacian = topo_laplacian


        self.done = False

    def get_state(self):
        # include locations visited into the state
        return np.concatenate((np.concatenate((self.locations, self.agents_destination_coordinates,
                                               self.agents_destination_coordinates[self.agent_taking_decision,
                                               :].reshape(1, 2)), axis=0).reshape(-1, 1),
                               self.nodes_visited.reshape(-1, 1)))

    def get_encoded_state(self):
        mask = self.get_mask()
        task_graph_nodes, task_graph_adjacency = self.generate_task_graph()
        agents_graph_nodes, agents_graph_adjacency = self.generate_agents_graph()
        state = {
            # 'location': self.locations,
            'depot': self.depot.reshape(1, 2),
            'mask': mask,
            # 'agents_destination_coordinates': self.agents_destination_coordinates,
            # 'agent_taking_decision_coordinates': self.agents_destination_coordinates[
            #                                      self.agent_taking_decision, :].reshape(1,
            #                                                                             2),
            # 'topo_laplacian':self.topo_laplacian,
            'task_graph_nodes': task_graph_nodes,
            # 'task_graph_adjacency':task_graph_adjacency,
            'agents_graph_nodes':agents_graph_nodes,
            # 'agents_graph_adjacency':agents_graph_adjacency,
            'nodes_visited':self.nodes_visited,
            'first_dec': self.first_dec,
            'agent_taking_decision': self.agent_taking_decision
        }

        return state

    def var_preprocess(self, adj, r):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized

    def get_topo_laplacian(self, data):
        active_tasks = ((data['nodes_visited'] == 0).nonzero())[0]
        X_loc = (data['task_graph_nodes'].numpy())[None,:]
        X_loc = X_loc[:, active_tasks[1:] - 1, :]
        # distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)[0]
        distance_matrix = torch.cdist(torch.tensor(X_loc), torch.tensor(X_loc),p=2)[0]

        adj_ = np.float32(distance_matrix < 0.3)

        dt = defaultdict(list)
        for i in range(adj_.shape[0]):
            n_i = adj_[i, :].nonzero()[0].tolist()

            dt[i] = n_i

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dt))
        adj_array = adj.toarray().astype(np.float32)
        var_laplacian = self.var_preprocess(adj=adj, r=2).toarray()

        secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat=adj_array, w_adj_mat=distance_matrix, k=3)

        reg_dgms = list()
        for i in range(len(secondorder_subgraph)):
            # print(i)
            tmp_reg_dgms = simplicial_complex_dgm(secondorder_subgraph[i])
            if tmp_reg_dgms.size == 0:
                reg_dgms.append(np.array([]))
            else:
                reg_dgms.append(np.unique(tmp_reg_dgms, axis=0))

        reg_dgms = np.array(reg_dgms)

        row_labels = np.where(var_laplacian > 0.)[0]
        col_labels = np.where(var_laplacian > 0.)[1]

        topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype=np.float32)

        for i in range(row_labels.shape[0]):
            tmp_row_label = row_labels[i]
            tmp_col_label = col_labels[i]
            tmp_wasserstin_dis = wasserstein(reg_dgms[tmp_row_label], reg_dgms[tmp_col_label])
            # if tmp_wasserstin_dis == 0.:
            #     topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / 1e-1
            #     topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / 1e-1
            # else:
            topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / (tmp_wasserstin_dis+1)
            topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / (tmp_wasserstin_dis+1)

        return topo_laplacian_k_2

    def step(self, action):
        # print("Time: ", self.time)
        # print("New action taken from the available list: ", action)
        action = self.active_tasks[action]
        # print("Actual action: ", action)
        self.episode_start = 0
        self.first_dec = False

        agent_taking_decision = self.agent_taking_decision  # id of the agent taking action
        current_location_id = self.current_location_id  # current location id of the robot taking decision

        self.total_length = self.total_length + 1

        reward = 0.0
        info = {}
        travel_distance = self.distance_matrix[current_location_id, action]
        self.agents_current_range[0, agent_taking_decision] -= travel_distance
        self.visited.append((action, self.agent_taking_decision))
        if action == self.depot_id:
            self.agents_current_payload[0, agent_taking_decision] = self.max_capacity
            self.agents_current_range[0, agent_taking_decision] = self.max_capacity
            self.nodes_visited[action] = 0
        if self.nodes_visited[action] != 1 and action != self.depot_id:

            distance_covered = self.total_distance_travelled + travel_distance
            self.total_distance_travelled = distance_covered
            self.agents_distance_travelled[agent_taking_decision] += travel_distance
            self.agents_current_payload[0, agent_taking_decision] -= self.location_demand[0, action]

            # update the the status of the node_visited that was chosen
            self.nodes_visited[action] = 1
            if self.time_deadlines[0, action] < torch.tensor(self.time):
                self.deadline_passed[0, action] = 1
            else:
                self.task_done[0, action] = 1

            # print(sum(self.nodes_visited))
            # reward = -travel_distance
            # reward = -travel_distance#1/(travel_distance*100 + 10e-5)
            self.total_reward += reward


            # else:
            # reward = 0.0

            # change destination of robot taking decision
        self.agents_next_location[agent_taking_decision] = action
        self.agents_prev_location[agent_taking_decision] = current_location_id
        self.agents_destination_coordinates[agent_taking_decision] = self.locations[action]
        self.agents_distance_to_destination[agent_taking_decision] = travel_distance
        self.agents_next_decision_time[agent_taking_decision] = self.time + travel_distance / self.agent_speed

        ## finding the agent which takes the next decision
        self.agent_taking_decision = np.argmin(self.agents_next_decision_time)
        self.current_location_id = self.agents_next_location[self.agent_taking_decision]
        self.time = self.agents_next_decision_time[self.agent_taking_decision]
        deadlines_passed_ids = (self.time_deadlines < torch.tensor(self.time)).nonzero()
        # print("Deadline passed: ", deadlines_passed_ids[:,1].T)
        if deadlines_passed_ids.shape[0] != 0:

            self.deadline_passed[0, deadlines_passed_ids[:,1]] = 1
            self.nodes_visited[deadlines_passed_ids[:, 1], 0] = 1
        # print("Active tasks before update: ", self.active_tasks)
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        # print("Active tasks after update: ", self.active_tasks)

        if sum(self.nodes_visited) == self.n_locations - 1:
            # reward = 1/(self.total_distance_travelled + 10e-5) - (self.total_length - self.n_locations+1)/self.n_locations
            # 1/(self.total_distance_travelled**2+ 10e-5)## change this with the distance travelled
            # self.total_reward += reward
            final_distance_to_depot = torch.cdist(torch.tensor(self.agents_destination_coordinates), torch.tensor(self.depot[None,:])).sum().item()
            if self.task_done.sum() == self.n_locations - 1:
                reward = -(self.total_distance_travelled +final_distance_to_depot)/ (1.41 * self.n_locations)
            else:
                reward = -((self.n_locations - 1) - self.task_done.sum())/((self.n_locations - 1))
            self.total_reward = reward
            self.done = True
            # if self.total_length == self.n_locations-1:
            #     reward = reward + 1
            info = {"is_success": self.done,
                    "episode": {
                        "r": self.total_reward,
                        "l": self.total_length
                    }
                    }


        # else:
        #     reward = 0  # -100
        #     self.total_reward += reward

            # Set placeholder for info

        # Return step information
        return self.get_encoded_state(), reward, self.done, info

    def get_mask(self):
        agent_taking_decision = self.agent_taking_decision
        mask = self.nodes_visited.copy()
        current_location_id = self.current_location_id
        if self.agents_current_payload[0, agent_taking_decision] == 0:
            mask[1:,0] = 1
            mask[0, 0] = 0
        elif current_location_id == self.depot_id:
            mask[0, 0] = 1
        else:
            unreachbles = (self.distance_matrix[0,:] + self.distance_matrix[current_location_id,:] > self.agents_current_range[0, agent_taking_decision].item()).nonzero()
            if unreachbles[0].shape[0] != 0:
                mask[unreachbles[1], 0] = 1
            mask = np.logical_or(mask, (self.deadline_passed.T).numpy()).astype(mask.dtype)
            if mask[1:,0].prod() == 1: # if no other feassible locations, then go to depot
                mask[0,0] = 0



        if mask.prod() != 0.0:
            mask[0,0] = 0
        return mask
        # masking:
        #   nodes visited - done
        #   capacity = 0 -> depot - done
        #   Range not sufficient to reach depot -> depot
        #   deadlines passed done
        #    if current location is depot, then mask the depot - done

    def generate_task_graph(self):

        # if self.active_tasks.shape == 0:
        #     print("Error....")
        locations = torch.tensor(self.locations)
        time_deadlines = (self.time_deadlines.T)
        location_demand = (self.location_demand.T)
        node_properties = torch.cat((locations, time_deadlines, location_demand), dim=1)
        node_properties = node_properties[1:, :] # excluding the depot
        node_properties = node_properties/node_properties.max(dim=0).values # normalizing
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1/(1+torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix*(distance_matrix>0).to(torch.float32) # setting diagonal elements as 0
        return node_properties, adjacency_matrix

    def generate_agents_graph(self):
        node_properties = torch.cat((torch.tensor(self.agents_destination_coordinates), self.agents_current_range.T, self.agents_current_payload.T, torch.tensor(self.agents_next_decision_time)), dim=1)
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1 / (1 + torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix * (distance_matrix > 0).to(torch.float32) # setting diagonal elements as 0
        return node_properties, adjacency_matrix

    # def generate_feasible_task_graph(self, current_node_properties, current_ids):
    #     pass



    def render(self):
        # Add visualization
        print(self.action_space)

    def reset(self):
        self.action_space = Discrete(1)
        self.locations = np.random.random((self.n_locations, 2))
        self.depot = self.locations[0, :]
        self.visited = []
        self.agents = []
        self.agent_taking_decision = 1
        self.agents_location = []
        self.agents_prev_location = np.zeros((self.n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((self.n_agents, 1), dtype=int)
        self.agents_distance_travelled = np.zeros((self.n_agents, 1))
        self.total_distance_travelled = 0.0
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((self.n_locations, 1))
        self.agents_distance_to_destination = np.zeros(
            (self.n_agents, 1))  # tracks the distance to destination from its current point for all robots

        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agent_speed = 0.01
        self.agents_next_decision_time = np.zeros((self.n_agents, 1))
        self.agents_destination_coordinates = np.ones((self.n_agents, 1)) * self.depot
        self.total_reward = 0.0
        self.total_length = 0
        self.first_dec = True
        self.agents_current_range = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_range
        self.agents_current_payload = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_capacity
        self.time_deadlines = (torch.tensor(np.random.random((1, self.n_locations))) * .3 + .7) * 200
        self.time_deadlines[0, 0] = 1000000
        self.location_demand = torch.ones((1, self.n_locations), dtype=torch.float32)
        self.task_done = torch.zeros((1, self.n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, self.n_locations), dtype=torch.float32)
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        # Reset the number of not-done tasks
        # self.unvisited = self.all_task
        self.done = False

        self.topo_laplacian = None
        state = self.get_encoded_state()
        # topo_laplacian = self.get_topo_laplacian(state)
        # state["topo_laplacian"] = topo_laplacian
        # self.topo_laplacian = topo_laplacian
        return state
        # Reset to depot location

