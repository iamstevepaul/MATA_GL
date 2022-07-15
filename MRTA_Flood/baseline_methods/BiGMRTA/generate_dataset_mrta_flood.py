"""
Author: Steve Paul 
Date: 7/4/22 """
import numpy as  np
import torch as th
def get_new_scenario(
        n_locations,
        n_robots):
    deadline = ((th.tensor(np.random.random((n_locations, 1))) * .3 + .7) * 100).numpy()
    location = np.random.random((n_locations, 2))*30
    rnd = np.random.random((n_locations, 1))
    depot = np.random.random((1, 2))*30
    dat = np.concatenate([location, rnd, deadline], axis = 1)
    data = {
        'taskData': dat,
        'depotLocation': depot
    }
    return data
