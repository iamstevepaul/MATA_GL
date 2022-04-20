"""
Author: Steve Paul 
Date: 3/20/22 """
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_locs = np.array([21,31,41,61,81,101])
n_agents = np.array([3,6])
m_factor = np.array([1,1,2,3,4,5])
results = {}
Exps =[4, 6, 7]

columns = ['reward', 'exp', 'scenario']
dataframe = []
for i in range(n_locs.shape[0]):
    n_loc = n_locs[i]
    m = m_factor[i]
    n_agent = n_agents*m
    for n in n_agent:
        comp = []
        for exp in Exps:
            filename = str(exp)+ "_" + str(n_loc) + "_" + str(n) + ".pkl"
            if exp == 4:
                exp_name = "CAP-TDA"
            elif exp == 6:
                exp_name = "CAP"
            elif exp == 7:
                exp_name = "MLP"
            scenario = str(n_loc) + "_" + str(n)
            with open(filename, "rb") as fl:
                data = pickle.load(fl)
                for dat in data:
                    dataframe.append([dat[0], exp_name, scenario])
                comp.append(data)

        comp1 = np.array(comp[0])
        comp2 = np.array(comp[1])
        dt = 0
        # plt.plot(comp)

df = pd.DataFrame(dataframe, columns = columns)
sns.boxplot(x = df['scenario'],
            y = df['reward'],
            hue = df['exp'],
            palette = 'husl')

plt.show()
