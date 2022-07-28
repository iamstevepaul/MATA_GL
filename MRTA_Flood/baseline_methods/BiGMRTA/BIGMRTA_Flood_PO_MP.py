"""
Author: Steve Paul 
Date: 7/28/22 """
## running the BIGMRTA on multiple prcessors
# !/usr/bin/env python

from multiprocessing import Process
from multiprocessing import Queue, Manager
import random

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



# calc_pi() - compute a number of monte carlo samples
# and count number of samples that are in the quadrant
# of a unit cirle. The final tally is placed on the
# results queue for reduction by the main process.

def calc_pi(pid, samples, results_queue):
    count = 0
    for i in range(0,samples):
        x=random.uniform(0.00, +1.00)
        y=random.uniform(0.00, +1.00)
        test=(x*x)+(y*y)
        if(test <= 1.00) :
            count = count + 1
    results_queue.put(count)


def simulation(start, batch_size, env_lists, nRobot, n_loc_test,total_rewards_list, total_tasks_done_list):
    maxRun = 1
    # env_lists_batch = env_lists[start:start + batch_size]
    for en in range(start, start+batch_size):
        env = env_lists[en]
    # for env in env_lists_batch:
        # env.envs[0].training = False

        for iRun in range(1, maxRun + 1):
            print("--BEGIN: " + str(iRun) + "--------------------------------------------------\n")
            # Read the CaseStudy data
            # data = scipy.io.loadmat('Data/FloodSim_DataSet_n'+str(nAllTasks)+'_run_'+str(iRun)+'.mat')
            data = env  # get_new_scenario(n_locations=nAllTasks, n_robots=nRobot)
            # print("Run using FloodSim_DataSet_n" + str(nAllTasks) + "_run_" + str(iRun) + "\n")

            taskDataNs = data['taskData']
            taskData = taskDataNs[taskDataNs[:, 3].argsort()]
            taskLocation = taskData[:, :2] * 100
            taskTime = taskData[:, -1] * 60
            depotLocation = data['depotLocation'] * 100  # [:,:]
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
            robotStateRecords = np.zeros((nRobot, nRobot, 8))
            robotState[:, 3] = Q
            robotStateRecords[:, :, 3] = Q
            robotState[:, 4] = Range
            robotStateRecords[:, :, 4] = Range
            robotStateRecords[:, :, 7] = 0.0
            robotVelocity = np.zeros((nRobot, 2))
            robotLocation = np.ones((nRobot, 2)) * depotLocation
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
                        if (robotStateRecords[iRobot, iRobot, 1] - t <= decTime):
                            if robotStateRecords[
                                iRobot, iRobot, 0] == 0:  # Returned to depot: refill payloads and reset range
                                robotStateRecords[iRobot, iRobot, 3] = Q
                                robotStateRecords[iRobot, iRobot, 4] = Range
                            else:
                                robotStateRecords[iRobot, iRobot, 3] = robotStateRecords[iRobot, iRobot, 3] - 1
                                robotStateRecords[iRobot, iRobot, 4] = robotStateRecords[iRobot, iRobot, 4] - \
                                                                       robotStateRecords[iRobot, iRobot, 2]
                            robotStateRecords[iRobot, iRobot, 5] = robotStateRecords[iRobot, iRobot, 5] + \
                                                                   robotStateRecords[iRobot, iRobot, 2]

                    for iRobot in range(nRobot):  # Robot take decisions
                        # Check is near to goal (<60 sec)
                        if (robotStateRecords[iRobot, iRobot, 1] - t <= decTime):
                            nCurrentTask = len(taskNodes)
                            tic()
                            prvLoc = int(robotStateRecords[iRobot, iRobot, 0])
                            if robotStateRecords[iRobot, iRobot, 3] > 0 and (
                                    robotStateRecords[iRobot, iRobot, 4] - distanceMatrix[prvLoc, 0] > 0):
                                nxtLoc, graphSize = getNextTask(t, iRobot, robotStateRecords[iRobot, :, :], robotNodes,
                                                                taskNodes,
                                                                distanceMatrix, timeMatrix, timeDeadline)
                            else:
                                nxtLoc = 0
                            tm = toc()
                            if loc[nxtLoc, 0] == loc[prvLoc, 0] and loc[nxtLoc, 1] == loc[prvLoc, 1]:
                                velocity = np.zeros((2))
                            else:
                                velocity = ((loc[nxtLoc] - loc[prvLoc]) / np.linalg.norm(
                                    loc[nxtLoc] - loc[prvLoc])) * Vavg
                            robotVelocity[iRobot, :] = velocity
                            robotStateRecords[iRobot, iRobot, 7] = t
                            tempRobotStatus[iRobot] = nxtLoc
                            robotStateRecords[iRobot, iRobot, 6] = robotStateRecords[iRobot, iRobot, 6] + tm
                            if isDebug:
                                print('{} -> {}; t={}'.format(robotNodes[iRobot], nxtLoc, tm))
                            robotHistory[robotNodes[iRobot]] = np.vstack((robotHistory[robotNodes[iRobot]],
                                                                          [t, tm, nCurrentTask, graphSize, nxtLoc,
                                                                           robotStateRecords[iRobot, iRobot, 3]]))

                            decisionHistory = np.vstack((decisionHistory,
                                                         [t, tm, graphSize, nxtLoc, iRobot]))

                    for iRobot in range(nRobot):  # Robot Communicate to inform about their decisions
                        if (robotStateRecords[iRobot, iRobot, 1] - t <= decTime):
                            nxtLoc = int(tempRobotStatus[iRobot])
                            if nxtLoc != 0:
                                if isDebug:
                                    print(prvLoc, nxtLoc, iRobot, taskNodes)
                                if taskNodes.count(nxtLoc) != 0:
                                    taskNodes.remove(nxtLoc)
                            prvLoc = int(robotStateRecords[iRobot, iRobot, 0])
                            robotStateRecords[iRobot, iRobot, 0] = nxtLoc
                            robotStateRecords[iRobot, iRobot, 1] = robotStateRecords[iRobot, iRobot, 1] + timeMatrix[
                                prvLoc, nxtLoc]
                            robotStateRecords[iRobot, iRobot, 2] = distanceMatrix[prvLoc, nxtLoc]

                    for iRobot1 in range(nRobot - 1):
                        for iRobot2 in range(iRobot1 + 1, nRobot):
                            if np.linalg.norm(robotLocation[iRobot2, :] - robotLocation[iRobot1, :]) < comm_thresh:
                                for rob in range(nRobot):
                                    if robotStateRecords[iRobot1, rob, 7] >= robotStateRecords[iRobot2, rob, 7]:
                                        robotStateRecords[iRobot2, rob, :] = robotStateRecords[iRobot1, rob, :]
                                    else:
                                        robotStateRecords[iRobot1, rob, :] = robotStateRecords[iRobot2, rob, :]
                else:
                    break
                robotLocation += robotVelocity * timeStep
            for iRobot in range(nRobot):  # Ensure all go back to depot
                if (robotStateRecords[iRobot, iRobot, 0] != 0):
                    nxtLoc = 0
                    prvLoc = int(robotStateRecords[iRobot, iRobot, 0])
                    robotStateRecords[iRobot, iRobot, 0] = nxtLoc
                    robotStateRecords[iRobot, iRobot, 1] = robotStateRecords[iRobot, iRobot, 1] + timeMatrix[
                        prvLoc, nxtLoc]
                    robotStateRecords[iRobot, iRobot, 2] = distanceMatrix[prvLoc, nxtLoc]
                    robotStateRecords[iRobot, iRobot, 5] = robotStateRecords[iRobot, iRobot, 5] + robotStateRecords[
                        iRobot, iRobot, 2]

            numTaskDone = nTask - len(taskNodes)
            totalCost = sum(robotState[:, 5])
            computationTimeWhole = np.mean(robotState[:, 6])
            reward = (numTaskDone - n_loc_test + 1) / (n_loc_test - 1)
            total_rewards_list[en] = reward
            total_tasks_done_list[en] = numTaskDone


# main program - fires off a bunch of processes to
# compute monte carlo samples and count how many are
# in the unit circle. The final tally from all processes
# is used to estimate the value of pi.
# if __name__ == "__main__":
    # extract num processes from SLURM
# Directory that you want to save results and outputs

isDebug = False

# nAllTasks = 1000
# nInitTasks = 100
problem = "MRTA_Flood"
task_type = "ND"

trained_model_n_loc = 51
trained_model_n_robots = 6
loc_test_multipliers = [0.5,1,2,5,10]
robot_test_multipliers = [0.5,1,2]
comm_thresh = 100
path =  "Data/" + problem + "/"
nproc = int(os.getenv('SLURM_NPROCS', '4'))
# nproc = 5

samples=15
if __name__ == "__main__":
    m = Manager()
    output_dir = "Results"
    # If folder doesn't exist, then create it.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    # the processes will place their individual tallies
    # in the queue when they finish.
    q = Queue()
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
            total_rewards_list = m.list(range(samples))
            total_tasks_done_list = m.list(range(samples))
            distance_list = []
    # create np processes. each one with invoke calc_pi()
    # and operate on a subset of the total number of samples.

            procs = []
            batch_size = int(samples/nproc)
            for i in range(0,nproc):
                start = int(i*batch_size)

                p = Process(target=simulation, args=(start, batch_size, env_lists, nRobot, n_loc_test, total_rewards_list, total_tasks_done_list))
                procs.append(p)

            # launch the processes. They will run in parallel.
            for p in procs:
                p.start()

            # collect results from each process
            # count = 0
            # for p in procs:
            #   count = count + q.get()

            # compute final estimate of pi.
            # area of unit circle in the quandrant
            # is: pi/4 = count/samples

            # halt the child processes
            for p in procs:
                p.join()

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
            result_path = output_dir+"/" + problem

            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            result_file = result_path+ "/" + problem + "_PO" + "_nloc_" + str(n_loc_test) \
                          + "_nrob_" + str(n_robots_test) + "_" + task_type + "_" + "BIGMRTA"
            mode = 0o755
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path, mode)
            with open(result_file, 'wb') as fl:
                pickle.dump(data, fl)
            fl.close()
            ft = 0