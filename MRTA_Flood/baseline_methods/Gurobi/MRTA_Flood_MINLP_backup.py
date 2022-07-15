"""
Author: Steve Paul 
Date: 7/2/22 """

# Gurobi based MINLP solution for MRTA-Flood problem
import numpy as np
from gurobipy import *
from scipy.spatial import distance_matrix

x_max = 1
x_min = 0
y_max = 1
y_min = 0
n_locations = 20
n_robots = 4
loc = np.random.random((n_locations, 2))
work_load = np.random.random((n_locations, 1))*10
time_deadline = np.random.randint(600, 1000, (n_locations, 1))
robot_start_loc = np.random.random((n_robots, 2))
robot_work_capacity = np.random.random((n_robots, 1))
speed = 0.01
dist_mat = distance_matrix(loc, loc)
time_mat = dist_mat/speed
total_dec_per_robot = n_locations

starting_distance_matrix = distance_matrix(loc, robot_start_loc)
starting_time_matrix = starting_distance_matrix/speed
robot_work_time = np.zeros((n_locations, n_robots))
for i in range(n_locations):
    for k in range(n_robots):
        robot_work_time[i,k] = work_load[i,0]/robot_work_capacity[k,0]

model = Model("MRTA_TAPTC")

x_start = model.addVars(n_robots, total_dec_per_robot, n_locations, vtype=GRB.BINARY, name="x_start")
x_end = model.addVars(n_robots, total_dec_per_robot, n_locations, vtype=GRB.BINARY, name="x_end")
dt = model.addVars(n_robots, total_dec_per_robot,n_locations, vtype=GRB.CONTINUOUS, name="dt")
task_time = model.addVars(n_locations, vtype=GRB.CONTINUOUS, name="task_time") # recheck

for i in range(n_locations): # each location will have one start point during any time (from dec 2)
    model.addConstr(quicksum(x_start[k,l,i] for k in range(n_robots) for l in range(total_dec_per_robot)) <= 1)
for k in range(n_robots):
    for i in range(n_locations):
        model.addConstr(x_start[k,0,i] == 0)
for k in range(n_robots):
    for l in range(1,total_dec_per_robot):
        model.addConstr(quicksum(x_start[k,l,i] for i in range(n_locations)) == 1)

for i in range(n_locations): # each location will have one end point during any time
    model.addConstr(quicksum(x_end[k,l,i] for k in range(n_robots) for l in range(total_dec_per_robot)) == 1)

for k in range(n_robots):
    for l in range(total_dec_per_robot):
        model.addConstr(quicksum(x_end[k,l,i] for i in range(n_locations)) == 1)

# for k in range(n_robots):
#     for l in range(1, total_dec_per_robot):
#         for i in range(n_locations):
#             model.addConstr(x_start[k,l,i] != x_end[k,l,i])

# add on constraint such that the current start point and previous end point of an agent are the same
for k in range(n_robots):
    for l in range(1, total_dec_per_robot):
        for i in range(n_locations):
            model.addConstr(x_start[k,l,i] == x_end[k,l-1,i])

for k in range(n_robots):
    for i in range(n_locations):
            model.addConstr(dt[k,0,i] == x_end[k,0,i]*(starting_time_matrix[i,k] + robot_work_time[i,k]))

for l in range(1, total_dec_per_robot):
    for k in range(n_robots):
        for i in range(n_locations):
            for j in range(n_locations):
                model.addQConstr(dt[k,l,j] == (x_start[k,l,i]*x_end[k,l,j])*(task_time[i] + time_mat[i,j] + robot_work_time[j,k]))


for i in range(n_locations):
    model.addConstr(
        quicksum(dt[k,l,i] for k in range(n_robots) for l in range(total_dec_per_robot)) == task_time[i]
    )
model.setObjective(
    quicksum((task_time[i] - time_deadline[i]) for i in range(n_locations)
             ), GRB.MINIMIZE
)

model.setParam('MIPFocus', 3)
model.setParam("TimeLimit", 3600.0)
model.setParam('NonConvex', 2)

model.optimize()

#todo ---- we need add one more dimension to account for the 'to' location during a transition


