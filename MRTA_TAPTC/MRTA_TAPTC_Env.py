"""
Author: Steve Paul 
Date: 6/15/22 """


import numpy as np
import gym
import time
from gym import Env
from collections import defaultdict
from gym.spaces import Discrete, MultiBinary, Box, Dict
import random
import matplotlib.pyplot as plt
import torch
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot


class MRTA_TAPTC_Env(Env):

    def __init__(self,
                 n_locations=100,
                 n_agents=2,
                 enable_dynamic_tasks = False,
                 n_initial_tasks = 30,
                 display = False,
                 enable_topological_features = False,
                 agents_info_exchange_distance_threshold=0.7
                 ):
        super(MRTA_TAPTC_Env, self).__init__()
        self.n_locations = n_locations
        self.action_space = Discrete(1)
        self.locations = np.random.random((n_locations, 2))
        self.visited = []
        self.n_agents = n_agents
        self.agents_prev_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_distance_travelled = np.zeros((n_agents, 1))
        self.total_distance_travelled = 0.0
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((n_locations, 1))
        self.n_locations = n_locations
        self.enable_dynamic_tasks = enable_dynamic_tasks
        self.agents_distance_to_destination = np.zeros(
            (n_agents, 1))  # tracks the distance to destination from its current point for all robots

        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agent_speed = 0.01
        self.agents_next_decision_time = np.zeros((n_agents, 1))
        self.agents_prev_decision_time = np.zeros((n_agents, 1))
        self.agents_destination_coordinates = np.random.random((self.n_agents, 2))
        self.total_reward = 0.0
        self.total_length = 0

        self.robot_work_capacity = torch.randint(1, 3, (self.n_agents, 1), dtype=torch.float).view(-1) / 100
        self.task_workload = torch.ones((self.n_locations, 1))*.2
        self.time_deadlines = (torch.tensor(np.random.random((1, n_locations)))*.3 + .7)*200
        self.task_done = torch.zeros((1, n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, n_locations), dtype=torch.float32)
        self.available_tasks = torch.zeros((n_locations, 1), dtype=torch.float32)

        if not self.enable_dynamic_tasks:
            n_initial_tasks = n_locations
        self.n_initial_tasks = n_initial_tasks
        self.available_tasks[0: n_initial_tasks, 0] = 1 # set the initial tasks available
        self.time_start = self.time_deadlines*(torch.rand((n_locations,1))*.5).T
        self.time_start[0,0:self.n_initial_tasks] = 0
        self.display = display
        self.enable_topological_features = enable_topological_features
        self.agents_state_record = torch.zeros((self.n_agents,self.n_agents, 6), dtype=torch.float32) # keeps record of
        # each agents info on other agents along with a time stamp. 7 - 6 (inclusind current dest id) state vars of agents and 1 for time stamp
        self.agents_state_record[:,:,-1] = -1 # starting as -1
        self.agents_current_coordinates = self.agents_destination_coordinates.copy()
        self.agents_previous_recorded_coordinates = self.agents_destination_coordinates.copy()
        self.agents_velocity = torch.zeros((self.n_agents, 1), dtype=torch.float32)

        self.agents_info_exchange_distance_threshold = agents_info_exchange_distance_threshold
        self.agents_record_nodes_visited = torch.zeros((self.n_agents,self.n_locations, 1), dtype=torch.float32)

        self.conflicts_count = 0

        self.task_graph_node_dim = self.generate_task_graph()[0].shape[1]
        self.agent_node_dim = self.generate_agents_graph()[0].shape[1]

        if self.enable_topological_features:
            self.observation_space = Dict(
                dict(
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    topo_laplacian=Box(low=0, high=1, shape=(n_locations,n_locations)),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations - 1, 5)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, 5)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,), dtype=int),
                ))
            self.topo_laplacian = None
            state = self.get_encoded_state()
            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        else:
            self.observation_space = Dict(
                dict(
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations,5)),
                    task_graph_adjacency=Box(low=0, high=1, shape=(n_locations, n_locations)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, 5)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,), dtype=int),
                ))
        self.distance = 0.0
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
        if self.enable_topological_features:
            state = {
                'mask': mask,
                'task_graph_nodes': task_graph_nodes,
                'topo_laplacian': self.topo_laplacian,
                'agents_graph_nodes':agents_graph_nodes,
                'agent_taking_decision': self.agent_taking_decision,
            }
        else:
            state = {
                'mask': mask,
                'task_graph_nodes': task_graph_nodes,
                'task_graph_adjacency':task_graph_adjacency,
                'agents_graph_nodes': agents_graph_nodes,
                'agent_taking_decision': self.agent_taking_decision,
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
        X_loc = (data['task_graph_nodes'].numpy())[None,:]
        distance_matrix = torch.cdist(torch.tensor(X_loc), torch.tensor(X_loc),p=2)[0]

        adj_ = np.float32(distance_matrix < 0.3)

        adj_ = adj_ * (self.available_tasks[1:, :].T).numpy()
        adj_ = adj_ * (self.available_tasks[1:, :]).numpy()

        dt = defaultdict(list)
        for i in range(adj_.shape[0]):
            n_i = adj_[i, :].nonzero()[0].tolist()

            dt[i] = n_i

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dt))
        adj_array = adj.toarray().astype(np.float32)
        var_laplacian = self.var_preprocess(adj=adj, r=2).toarray()

        secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat=adj_array, w_adj_mat=distance_matrix, k=2)

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
        # print("Actual action: ", action)
        # print(self.agent_taking_decision, action)




        agent_taking_decision = self.agent_taking_decision  # id of the agent taking action
        # print("Agent taking decision:", agent_taking_decision)
        current_location_id = self.current_location_id  # current location id of the robot taking decision

        if self.nodes_visited[action] == 1:

            self.conflicts_count += 1 # if the chosen task was already visited (according to ground truth, then increase the conflict count)

            self.agents_record_nodes_visited[agent_taking_decision, action] = 1

        self.total_length = self.total_length + 1

        reward = 0.0
        info = {}
        travel_distance = self.distance_matrix[current_location_id, action].copy()
        self.agents_prev_decision_time[agent_taking_decision, 0] = self.time
        self.visited.append((action, self.agent_taking_decision))
        work_time = self.task_workload[action,0]/self.robot_work_capacity[agent_taking_decision]
        if self.nodes_visited[action] != 1:

            distance_covered = self.total_distance_travelled + travel_distance
            self.total_distance_travelled = distance_covered
            self.agents_distance_travelled[agent_taking_decision] += travel_distance

            # update the  status of the node_visited that was chosen
            self.nodes_visited[action] = 1
            self.agents_record_nodes_visited[agent_taking_decision, action] = 1
            if self.time_deadlines[0, action] < torch.tensor(self.time + work_time):
                self.deadline_passed[0, action] = 1
            else:
                self.task_done[0, action] = 1

            self.total_reward += reward


        # update destination of robot taking decision
        self.agents_next_location[agent_taking_decision] = action
        self.agents_prev_location[agent_taking_decision] = current_location_id
        self.agents_destination_coordinates[agent_taking_decision] = self.locations[action].copy()
        self.agents_distance_to_destination[agent_taking_decision] = travel_distance
        self.agents_next_decision_time[agent_taking_decision] = self.time+ work_time + travel_distance / self.agent_speed

        # Here we update the self state record for the robot which made a decision
        self.agent_taking_decision_self_record_update()

        if self.display:
            self.render(action)

        ## finding the agent which takes the next decision
        self.agent_taking_decision = np.argmin(self.agents_next_decision_time)
        self.current_location_id = self.agents_next_location[self.agent_taking_decision][0].copy()

        prev_time = self.time

        self.time = self.agents_next_decision_time[self.agent_taking_decision][0].copy()
        vec = (self.agents_destination_coordinates - self.agents_previous_recorded_coordinates).copy()
        vec_unit = (vec / np.linalg.norm(vec, axis=1).reshape((self.n_agents, 1))).copy()
        where_are_NaNs = np.isnan(vec_unit)
        vec_unit[where_are_NaNs] = 0.0
        agents_velocity = vec_unit * self.agent_speed
        displacement = agents_velocity * (self.time - prev_time)
        self.agents_previous_recorded_coordinates = self.agents_current_coordinates.copy()
        self.agents_current_coordinates = self.agents_current_coordinates.copy() + displacement
        self.agents_velocity = agents_velocity



        self.agents_information_exchange(prev_time)# exchange of infrmation updated

        # print(self.agents_current_coordinates)
        # print("Agent taking decision coordinate:", self.agents_current_coordinates[self.agent_taking_decision])
        # print("Current location:", self.locations[self.current_location_id])


        deadlines_passed_ids = (self.time_deadlines < torch.tensor(self.time)).nonzero()
        # print("Deadline passed: ", deadlines_passed_ids[:,1].T)
        if deadlines_passed_ids.shape[0] != 0:
            # print("Some deadlines passed: ", deadlines_passed_ids[:,1])
            self.deadline_passed[0, deadlines_passed_ids[:,1]] = 1
            self.nodes_visited[deadlines_passed_ids[:, 1], 0] = 1
            # print("nodes visisted: ", self.nodes_visited)
            self.agents_record_nodes_visited[:, deadlines_passed_ids[:, 1], 0] = 1 # updating the state record of all the agents when some of the deadlines are passed

        # print("Active tasks after update: ", self.active_tasks)
        # print(sum(self.nodes_visited))

        self.available_tasks = (self.time_start <= self.time).to(torch.float32).T # making new tasks available
        # print(self.nodes_visited.T)
        # print(sum(self.nodes_visited) == self.n_locations)
        # print(sum(self.nodes_visited))
        # print(self.n_locations)
        if sum(self.nodes_visited) == self.n_locations:
            reward = -(self.n_locations - self.task_done.sum())/(self.n_locations)
            self.total_reward = reward
            self.done = True

            info = {"is_success": self.done,
                    "episode": {
                        "r": self.total_reward,
                        "l": self.total_length
                    }
                    }

        return self.get_encoded_state(), reward, self.done, info

    def agents_information_exchange(self, prev_time):
        # find the time intervals
        # find the time of minimum distance during that time period
        # if dt is negative, then use the starting time distance
        # if dt is greater than end time, use end time distance
        # else use the distance which is between start and end time
        # during information exchange:
            # Sender sends the infor to agents which was in its range. Time stamp of others are not changed,
            # self time stamp is updated
            # while receiving info the update happens only for latest time stamp

        for i in range(self.n_agents-1):
            for j in range(i+1, self.n_agents):
                x1 = self.agents_previous_recorded_coordinates[i]
                x2 = self.agents_previous_recorded_coordinates[j]
                v1 = self.agents_velocity[i]
                v2 = self.agents_velocity[j]
                dt = prev_time + self.find_closest_dist_time(x1,x2,v1,v2)

                if dt <= prev_time:
                    # dt = prev_time
                    x1_new = self.agents_previous_recorded_coordinates[i]
                    x2_new = self.agents_previous_recorded_coordinates[j]
                elif dt > prev_time and dt <= self.time:
                    x1_new = x1 + v1*(dt - prev_time)
                    x2_new = x2 + v2*(dt - prev_time)
                elif dt > self.time:
                    # dt = self.time
                    x1_new = self.agents_current_coordinates[i]
                    x2_new = self.agents_current_coordinates[j]


                min_distance = np.linalg.norm(x2_new - x1_new)

                if min_distance < self.agents_info_exchange_distance_threshold:
                    # print("information exchange happens")
                    states_time_i = self.agents_state_record[i,:,-1]
                    states_time_j = self.agents_state_record[j, :, -1]
                    # Updating the state record based on time stamp
                    ids_1 = (states_time_i > states_time_j).nonzero()
                    ids_2 = (states_time_i < states_time_j).nonzero()
                    if ids_1.shape[0] > 0:
                        self.agents_state_record[j, ids_1[:,0], :] = self.agents_state_record[i, ids_1[:,0], :].clone()
                        # if
                        self.agents_record_nodes_visited[j, self.agents_state_record[i, ids_1[:,0], 0].to(torch.int64)] = 1
                        # self.agents_record_nodes_visited[j, 0] = 0

                    if ids_2.shape[0] > 0:
                        self.agents_state_record[i, ids_2[:,0], :] = self.agents_state_record[j, ids_2[:,0], :].clone()
                        self.agents_record_nodes_visited[i, self.agents_state_record[j, ids_2[:, 0], 0].to(torch.int64)] = 1
                        # self.agents_record_nodes_visited[i, 0] = 0
                    # print("State record of i:", self.agents_state_record[i, :, :])
                    # print("State record of j:", self.agents_state_record[j, :, :])
                    # print("Nodes visited of i:",self.agents_record_nodes_visited[i,:])
                    # print("Nodes visited of j:", self.agents_record_nodes_visited[j, :])

    def agent_taking_decision_self_record_update(self):
        #whev an agent takes a decision, self state update is done
        max_time = self.time_deadlines[0,self.available_tasks.nonzero()[:,0]][1:].max()
        new_state = torch.cat((
            torch.tensor(self.agents_next_location[self.agent_taking_decision]),
            torch.tensor(self.agents_destination_coordinates[self.agent_taking_decision]),
            # torch.tensor([self.agents_current_range[0, self.agent_taking_decision]/self.max_range]),
            # torch.tensor([self.agents_current_payload[0, self.agent_taking_decision]/self.max_capacity]),
            self.robot_work_capacity[self.agent_taking_decision].unsqueeze(dim=0),
            torch.tensor(self.agents_next_decision_time[self.agent_taking_decision]/max_time),
            torch.tensor([self.time])), dim=0)
        self.agents_state_record[self.agent_taking_decision, self.agent_taking_decision] = new_state

    def find_closest_dist_time(self, x1,v1,x2,v2):
        # derivative equation for the time at which the closest distance happens given the starting locations and velocity
        delta_t = (-(x2[0] - x1[0])*(v2[0] - v1[0]) -(x2[1] - x1[1])*(v2[1] - v1[1]))/((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
        return delta_t

    def get_mask(self):

        agent_taking_decision = self.agent_taking_decision
        mask = self.agents_record_nodes_visited[agent_taking_decision].numpy().copy() # the node visited record for the agent taking decision is being used
        mask = np.logical_or(mask, (self.deadline_passed.T).numpy()).astype(mask.dtype)
        mask = mask*(self.available_tasks).numpy() # masking unavailable tasks
        return mask

    def generate_task_graph(self):

        locations = torch.tensor(self.locations)
        time_deadlines = (self.time_deadlines.T).clone()
        # location_demand = (self.location_demand.T).clone()
        deadlines_passed = (self.deadline_passed.T).clone()
        workload = self.task_workload
        node_properties = torch.cat((locations, time_deadlines, workload, deadlines_passed), dim=1)
        # node_properties = node_properties[1:, :] # excluding the depot
        node_properties[:, 0:3] = node_properties[:, 0:3]/node_properties[:, 0:3].max(dim=0).values # normalizing all except deadline_passed
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1/(1+torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix*(distance_matrix>0).to(torch.float32) # setting diagonal elements as 0
        node_properties = node_properties[:,:]*self.available_tasks[:,:] # masking the unavailable tasks
        adjacency_matrix = adjacency_matrix*(self.available_tasks[:,:].T)
        adjacency_matrix = adjacency_matrix*self.available_tasks[:,:]
        return node_properties, adjacency_matrix

    def generate_agents_graph(self):
        # node_properties = torch.cat((torch.tensor(self.agents_destination_coordinates), self.agents_current_range.T, self.agents_current_payload.T, torch.tensor(self.agents_next_decision_time)), dim=1)
        node_properties = self.agents_state_record[self.agent_taking_decision, :, 1:].clone()
        node_properties[:, -1] = (self.time - node_properties[:, -1]) / (1 + self.time)
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1 / (1 + torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix * (distance_matrix > 0).to(torch.float32) # setting diagonal elements as 0
        return node_properties, adjacency_matrix

    def render(self, action):

        pass




    def reset(self):
        self.action_space = Discrete(1)
        self.locations = np.random.random((self.n_locations, 2))
        self.visited = []
        self.agent_taking_decision = 1
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
        self.agents_prev_decision_time = np.zeros((self.n_agents, 1))
        self.agents_destination_coordinates = np.random.random((self.n_agents, 2))
        self.total_reward = 0.0
        self.total_length = 0
        self.robot_work_capacity = torch.randint(1, 3, (self.n_agents, 1), dtype=torch.float).view(-1) / 100
        self.task_workload = torch.ones((self.n_locations, 1))*.2
        self.time_deadlines = (torch.tensor(np.random.random((1, self.n_locations))) * .3 + .7) * 200
        self.time_deadlines[0, 0] = 1000000 # large number for depot,
        self.task_done = torch.zeros((1, self.n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, self.n_locations), dtype=torch.float32)
        # Reset the number of not-done tasks
        self.done = False

        if not self.enable_dynamic_tasks: # this conditional might be unnecessary
            n_initial_tasks = self.n_locations
        else:
            n_initial_tasks = self.n_initial_tasks
        self.n_initial_tasks = n_initial_tasks
        self.available_tasks[0: n_initial_tasks, 0] = 1 # set the initial tasks available
        self.time_start = self.time_deadlines*(torch.rand((self.n_locations,1))*.5).T
        self.time_start[0,0:self.n_initial_tasks] = 0

        self.agents_state_record = torch.zeros((self.n_agents,self.n_agents, 6),
                                               dtype=torch.float32)  # keeps record of each agents info on other agents along with a time stamp
        self.agents_state_record[:, :, -1] = -1  # starting as -1
        self.agents_current_coordinates = self.agents_destination_coordinates.copy()
        self.agents_previous_recorded_coordinates = self.agents_destination_coordinates.copy()
        self.agents_velocity = torch.zeros((self.n_agents, 1), dtype=torch.float32)

        self.agents_record_nodes_visited = torch.zeros((self.n_agents, self.n_locations, 1), dtype=torch.float32)

        self.conflicts_count = 0
        state = self.get_encoded_state()
        if self.enable_topological_features:
            self.topo_laplacian = None

            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        return state
        # Reset to depot location

