"""
Author: Steve Paul 
Date: 7/2/22 """

# Gurobi based MINLP solution for MRTA-Flood problem
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import distance as dist

from gurobipy import *

import scipy.io

import pickle

from time import time

from scipy.spatial import distance_matrix


tics = []

def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop()


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


            maxRun = 1
            for iRun in range(1, maxRun + 1):
                print("--BEGIN: " + str(iRun) + "--------------------------------------------------\n")
                # Read the CaseStudy data
                # data = scipy.io.loadmat('Data/FloodSim_DataSet_n'+str(nAllTasks)+'_run_'+str(iRun)+'.mat')
                data = env  # get_new_scenario(n_locations=nAllTasks, n_robots=nRobot)
                print("Run using FloodSim_DataSet_n" + str(nAllTasks) + "_run_" + str(iRun) + "\n")

                taskDataNs = data['taskData']
                taskData = taskDataNs[taskDataNs[:, 3].argsort()]
                taskLocation = taskData[:, :2] * 100
                taskTime = taskData[:, -1] * 60
                depotLocation = data['depotLocation'] * 100  # [:,:]
                loc = np.vstack((depotLocation, taskLocation))
                ## Alg parameters
                Q = 5  # Robot capacity (max number of payloads)
                Range = 140  # flight range in km
                Vavg = 1.21  # 40 km/h = 2/3 km/min

                ## Load data

                nTask = nAllTasks # np.shape(taskLocation)[0]
                nRoute = int(np.floor(nTask / nRobot)) + 2  # Number of maximum full trip

                V = np.arange(1, nTask + 1)
                Vp = np.hstack((0, V, nTask + 1))
                Vin = np.hstack((V, nTask + 1))
                Vout = np.hstack((0, V))
                S = np.arange(0, nRoute)
                R = np.arange(0, nRobot)

                n = np.size(V)
                ne = np.size(Vp)
                nr = np.size(S)
                m = np.size(R)
                data = np.vstack((depotLocation, taskLocation, depotLocation))

                C = dist.cdist(data, data, metric='euclidean')
                t = C / Vavg
                q = np.ones(ne)
                q[0] = 0
                q[-1] = 0

                tau = np.hstack((np.array([0]), taskTime, np.array([0])))

                print("V: {}".format(V))
                print("Vp: {}".format(Vp))
                print("S: {}".format(S))
                print("R: {}".format(R))
                print("tau: {}".format(tau))
                print("C: {}".format(C))

                tic()
                # Create a new model
                model = Model("MultipleTrip")

                ## Add variables
                # Arc Xijr
                x = model.addVars(ne, ne, m, nr, vtype=GRB.BINARY, name="x")
                # Route yikr
                y = model.addVars(ne, m, nr, vtype=GRB.BINARY, name="y")

                ## Set objective function
                pairIJ = [(i, j) for j in Vp for i in Vp]
                pairIS = [(i, s) for s in S for i in V]
                pairIJS = [(i, j, s) for s in S for j in Vp for i in Vp]
                pairRS = [(r, s) for s in S for r in R]
                pairIRS = [(i, r, s) for s in S for r in R for i in V]
                pairIJRS = [(i, j, r, s) for s in S for r in R for j in Vp for i in Vp]

                obj = quicksum(y[i, r, s] / (s + 1) for i, r, s in pairIRS)  # + quicksum(u[r,s]/(s+1) for r,s in pairRS)
                model.setObjective(obj, GRB.MAXIMIZE)

                # Take out some arcs
                for i in V:
                    for s in S:
                        for r in R:
                            # m.addConstr(x[0,i,s] == y[i,s])
                            model.addConstr(x[i, 0, r, s] == 0)
                            model.addConstr(x[ne - 1, i, r, s] == 0)
                            model.addConstr(x[i, i, r, s] == 0)
                            # model.addConstr(x[0,ne-1,r,s] == 0)
                for s in S:
                    for r in R:
                        model.addConstr(x[ne - 1, 0, r, s] == 1)

                for i in V:
                    for s in S:
                        for r in R:
                            # model.addConstr(quicksum(x[i,j,s] for j in Vin if i != j) == y[i,s])
                            # model.addConstr(quicksum(x[i,j,s] for j in Vin if i != j) == quicksum(y[i,r,s] for r in R))
                            model.addConstr(quicksum(x[i, j, r, s] for j in Vin if i != j) == y[i, r, s])

                for h in V:
                    for s in S:
                        for r in R:
                            model.addConstr(quicksum(x[i, h, r, s] for i in Vout if i != h) - quicksum(
                                x[h, j, r, s] for j in Vin if h != j) == 0)

                for i in V:
                    model.addConstr(quicksum(y[i, r, s] for r, s in pairRS) <= 1)
                    model.addConstr(quicksum(y[i, r, s] for r, s in pairRS) <= quicksum(x[0, j, r, s] for j, r, s in pairIRS))

                for r in R:
                    for s in S:
                        model.addConstr(y[0, r, s] == 1)
                        model.addConstr(y[ne - 1, r, s] == 1)

                for i in Vout:
                    for j in Vin:
                        if i != j:
                            model.addConstr(quicksum(x[i, j, r, s] for r, s in pairRS if i != j) <= 1)

                for s in S:
                    for r in R:
                        model.addConstr(quicksum(q[i] * y[i, r, s] for i in V) <= Q)

                for s in S:
                    for r in R:
                        model.addConstr(quicksum(C[i, j] * x[i, j, r, s] for i, j in pairIJ if i != j) <= Range)

                for s in range(nr - 1):  # as we compare y[i,r,s+1]
                    for r in R:
                        for ip in V:
                            pairIJr = [(i, j, rp) for rp in range(s) for j in Vp for i in Vp]
                            # t_0^(s+1) = t_(n+1)^s = quicksum(t[i,j]*x[i,j,r,rp] for i,j,rp in pairIJr)
                            model.addConstr(quicksum(t[i, j] * x[i, j, r, rp] for i, j, rp in pairIJr) >= tau[ip] * y[ip, r, s + 1])

                # Solve model
                tic()
                model.optimize()
                computationTimeSolve = toc()  # Measure time
                computationTimeWhole = toc()  # Measure time
                # Solution quality statistics
                model.printQuality()
                model.printAttr('X')
                # Statistics for model Unnamed
                model.printStats()
                print(computationTimeWhole)
                # m.getVars()

                # In[17]:

                ## Post-proc results
                sol = model.getVars()
                Xopt = np.zeros((ne, ne, m, nr))
                l = 0
                for i in Vp:
                    for j in Vp:
                        for r in R:
                            for s in S:
                                if sol[l].x == 1:
                                    # print("Solution {}, x[{},{},{}]".format(sol[l], i,j,s))
                                    Xopt[i, j, r, s] = 1
                                l = l + 1
                costPerRobot = np.zeros((m, 1))
                for r in R:
                    dummy = 0
                    for i in Vp:
                        for j in Vp:
                            for s in S:
                                dummy = dummy + C[i, j] * Xopt[i, j, r, s]
                    costPerRobot[r] = dummy

                totalCost = np.sum(costPerRobot)
                numTaskDone = model.getAttr('ObjVal')
                results = {'Xopt': Xopt, 'nRobot': nRobot, 'nTask': nTask,
                           'iRun': iRun, 'numTaskDone': numTaskDone, 'objVal': numTaskDone, 'costPerRobot': costPerRobot,
                           'totalCost': totalCost, 'computationTime': [computationTimeSolve, computationTimeWhole]}
                fileName = 'Results/CentralizedResults_m' + str(nRobot) + "_n" + str(nTask) + "_" + str(iRun)
                with open(fileName + '.pickle', 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("--END: " + str(iRun) + "--------------------------------------------------\n")

                model.write(fileName + ".mst")

                enablePlotting = False
                if enablePlotting:
                    plt.plot(d[0, 0], d[0, 1], 'o')
                    for i in V:
                        plt.plot(d[i, 0], d[i, 1], 's')
                        plt.annotate(str(i), (d[i, 0], d[i, 1]))

                    colorCode = ["s", "b", "g", "m", "c"]
                    for i in Vp:
                        for j in Vp:
                            for r in R:
                                for s in S:
                                    if Xopt[i, j, r, s] == 1:
                                        # plt.arrow(d[i,0], d[i,1], d[j,0], d[j,1], shape='full', lw=0, length_includes_head=True, head_width=.05)
                                        plt.plot([d[i, 0], d[j, 0]], [d[i, 1], d[j, 1]], color=colorCode[r])
                                        plt.annotate(str(r) + str(s), (d[i, 0], d[i, 1] * 1.05))

                    plt.show()

