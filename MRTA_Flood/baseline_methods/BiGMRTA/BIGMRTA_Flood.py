"""
Author: Steve Paul 
Date: 7/4/22 """

# !/usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import scipy.io
import pickle
import networkx as nx
from time import time
from generate_dataset_mrta_flood import get_new_scenario
from bigmrta import tic, toc, getNextTask, getParameters

# Directory that you want to save results and outputs
output_dir = "Results"
# If folder doesn't exist, then create it.
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

isDebug = False
maxRun = 1
# nAllTasks = 1000
# nInitTasks = 100
problem = "MRTA_Flood"
task_type = "ND"

trained_model_n_loc = 51
trained_model_n_robots = 6
loc_test_multipliers = [0.5,1,2,5,10]
robot_test_multipliers = [0.5,1,2]
path =  "Data/" + problem + "/"
for loc_mult in loc_test_multipliers:
    for rob_mult in robot_test_multipliers:
        n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots) + 1
        n_loc_test = int(trained_model_n_loc*loc_mult)


        nAllTasks = n_loc_test
        nRobot = n_robots_test
        file_name = path + problem \
                    + "_nloc_" + str(n_loc_test) \
                    + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
        with open(file_name, 'rb') as fl:
            env_lists = pickle.load(fl)
        fl.close()
        total_rewards_list = []
        distance_list = []
        total_tasks_done_list = []

        for env in env_lists:
            # env.envs[0].training = False

            for iRun in range(1, maxRun + 1):
                print("--BEGIN: " + str(iRun) + "--------------------------------------------------\n")
                # Read the CaseStudy data
                # data = scipy.io.loadmat('Data/FloodSim_DataSet_n'+str(nAllTasks)+'_run_'+str(iRun)+'.mat')
                data = env#get_new_scenario(n_locations=nAllTasks, n_robots=nRobot)
                print("Run using FloodSim_DataSet_n" + str(nAllTasks) + "_run_" + str(iRun) + "\n")

                taskDataNs = data['taskData']
                taskData = taskDataNs[taskDataNs[:, 3].argsort()]
                taskLocation = taskData[:, :2]*100
                taskTime = taskData[:, -1]*60
                depotLocation = data['depotLocation'] *100 # [:,:]
                loc = np.vstack((depotLocation, taskLocation))

                ## Alg parameters
                [Q, Range, Vavg, timeMax, timeStep, decTime, letancyTime] = getParameters()

                nTask = np.shape(taskLocation)[0]

                distanceMatrix = dist.cdist(loc, loc, metric='euclidean')
                timeMatrix = distanceMatrix / Vavg
                timeDeadline = np.hstack((np.array([0]), taskTime))

                robotNodes = []
                for i in range(nRobot):
                    robotNodes = np.append(robotNodes, 'r' + str(i + 1))

                taskNodes = list(np.arange(1, nTask + 1))

                robotState = np.zeros((nRobot, 7))
                robotState[:, 3] = Q
                robotState[:, 4] = Range
                coefRTask = 4
                # robotState - 0:2 -> Next, 3-6: Current
                #    0: index of current active task (mission),
                #    1: time when achieve its current active task (mission),
                #    2: Ditance travelled to finish the current task
                #    3: Current Remained Payload,
                #    4: Remained Range
                #    5: Overall distance travelled
                #    6: Overall computation time
                tempRobotStatus = np.zeros(nRobot)

                robotHistory = {'r1': []}
                decisionHistory = [[-1, -1, -1, -1, -1], ]
                for robotNode in robotNodes:
                    robotHistory[robotNode] = [[0, 0, len(taskNodes), 0, 0,
                                                Q], ]  # time, computing Time, Num of Task, Graph Size, Next Task, Remained Payload

                ## Simulation
                number_steps = int((timeMax + 1) / timeStep)
                for t in np.linspace(0, timeMax, number_steps):
                    if t % 10 == 0 or isDebug:
                        print(t)
                    if len(taskNodes) > 0:
                        for iRobot in range(nRobot):  # Communicate to update their status
                            if isDebug:
                                print(iRobot)
                            # Check is near to goal (<60 sec)
                            if (robotState[iRobot, 1] - t <= decTime):
                                if robotState[iRobot, 0] == 0:  # Returned to depot: refill payloads and reset range
                                    robotState[iRobot, 3] = Q
                                    robotState[iRobot, 4] = Range
                                else:
                                    robotState[iRobot, 3] = robotState[iRobot, 3] - 1
                                    robotState[iRobot, 4] = robotState[iRobot, 4] - robotState[iRobot, 2]
                                robotState[iRobot, 5] = robotState[iRobot, 5] + robotState[iRobot, 2]

                        for iRobot in range(nRobot):  # Robot take decisions
                            # Check is near to goal (<60 sec)
                            if (robotState[iRobot, 1] - t <= decTime):
                                nCurrentTask = len(taskNodes)
                                tic()
                                prvLoc = int(robotState[iRobot, 0])
                                if robotState[iRobot, 3] > 0 and (robotState[iRobot, 4] - distanceMatrix[prvLoc, 0] > 0):
                                    nxtLoc, graphSize = getNextTask(t, iRobot, robotState, robotNodes, taskNodes,
                                                                    distanceMatrix, timeMatrix, timeDeadline)
                                else:
                                    nxtLoc = 0
                                tm = toc()
                                tempRobotStatus[iRobot] = nxtLoc
                                robotState[iRobot, 6] = robotState[iRobot, 6] + tm
                                if isDebug:
                                    print('{} -> {}; t={}'.format(robotNodes[iRobot], nxtLoc, tm))
                                robotHistory[robotNodes[iRobot]] = np.vstack((robotHistory[robotNodes[iRobot]],
                                                                              [t, tm, nCurrentTask, graphSize, nxtLoc,
                                                                               robotState[iRobot, 3]]))

                                decisionHistory = np.vstack((decisionHistory,
                                                             [t, tm, graphSize, nxtLoc, iRobot]))

                        for iRobot in range(nRobot):  # Robot Communicate to inform about their decisions
                            if (robotState[iRobot, 1] - t <= decTime):
                                nxtLoc = int(tempRobotStatus[iRobot])
                                if nxtLoc != 0:
                                    if isDebug:
                                        print(prvLoc, nxtLoc, iRobot, taskNodes)
                                    taskNodes.remove(nxtLoc)
                                prvLoc = int(robotState[iRobot, 0])
                                robotState[iRobot, 0] = nxtLoc
                                robotState[iRobot, 1] = robotState[iRobot, 1] + timeMatrix[prvLoc, nxtLoc]
                                robotState[iRobot, 2] = distanceMatrix[prvLoc, nxtLoc]
                    else:
                        break
                for iRobot in range(nRobot):  # Ensure all go back to depot
                    if (robotState[iRobot, 0] != 0):
                        nxtLoc = 0
                        prvLoc = int(robotState[iRobot, 0])
                        robotState[iRobot, 0] = nxtLoc
                        robotState[iRobot, 1] = robotState[iRobot, 1] + timeMatrix[prvLoc, nxtLoc]
                        robotState[iRobot, 2] = distanceMatrix[prvLoc, nxtLoc]
                        robotState[iRobot, 5] = robotState[iRobot, 5] + robotState[iRobot, 2]

                numTaskDone = nTask - len(taskNodes)
                totalCost = sum(robotState[:, 5])
                computationTimeWhole = np.mean(robotState[:, 6])
                reward = (numTaskDone - n_loc_test+1)/(n_loc_test-1)
                total_rewards_list.append(reward)
                total_tasks_done_list.append(numTaskDone)
        total_rewards_array = np.array(total_rewards_list)
        total_tasks_done_array = np.array(total_tasks_done_list)
        data = {
            "problem": problem,
            "n_locations": n_loc_test,
            "n_robots": n_robots_test,
            "dynamic_task": False,
            "policy": "BIGMRTA",
            "total_tasks_done": total_tasks_done_array,
            "total_rewards": total_rewards_array,
        }
        result_path = "../../Results/" + problem + "/"

        result_file = result_path + problem + "_nloc_" + str(n_loc_test) \
                      + "_nrob_" + str(n_robots_test) + "_" + task_type + "_" + "BIGMRTA"
        mode = 0o755
        # if not os.path.exists(result_path):
        #     os.makedirs(result_path, mode)
        with open(result_file, 'wb') as fl:
            pickle.dump(data, fl)
        fl.close()

                # print('Results:')
                # print('Task Done = {}, Total Cost = {}, Total Computing Time (average across robots): {}'.format(numTaskDone, totalCost, computationTimeWhole))
                #
                # results = {'nRobot': nRobot, 'nTask': nTask, 'iRun': iRun, 'numTaskDone': numTaskDone, 'objVal': numTaskDone, 'decisionHistory': decisionHistory, 'totalCost': totalCost, 'computationTime': computationTimeWhole, 'robotState': robotState, 'robotHistory': robotHistory}
                # fileName = output_dir + '/DecMataResults_hungarian_m'+str(nRobot)+"_n"+str(nTask)+"_"+str(iRun)
                # with open(fileName+'.pickle', 'wb') as handle:
                #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # print("--END: "+str(iRun)+"--------------------------------------------------\n")

