
# class Action(object):
#     def __init__(self,heat_source):
#         self.source = heat_source

import numpy as np
from  .Room import Room, HeatSource
from matplotlib import pyplot as plt

class HeatEnv(object):
    """
    Args:
        boundary: list of HeatSources
        source of changement of temperature
    """
    metadata = {}

    def __init__(self,room_shape=(16,16),T_ideal=23):
        self.T_ideal = T_ideal

        self.room = Room(image=15*np.random.rand(room_shape[0],room_shape[1]))
        hsrc0 = HeatSource(T=20,x0=4,x1=12,y0=5,y1=7,name='radiator')
        hsrc1 = HeatSource(T=10, x0=6, x1=10, y0=1, y1=2, name='window')
        self.room.add_heat_source(hsrc0)
        self.room.add_heat_source(hsrc1)


    def step(self, a):
        if a ==1:
            self.room.heat_sources[0].T += 1.
            if self.room.heat_sources[0].T>50:
                self.room.heat_sources[0].T = 50
        elif a==2:
            self.room.heat_sources[0].T += -1.
            if self.room.heat_sources[0].T<10:
                self.room.heat_sources[0].T = 10

        _ = self.room.propagate(dt=0.2,dx=1.,dy=1.,n_steps=1000)

        assert(len(self.room.image.shape)==2)
        reward = 0.5*np.mean( np.exp(-np.sort(np.abs(self.room.image-self.T_ideal).reshape(-1))[::50]) )-0.5*np.abs(np.median(self.room.image)-self.T_ideal)/self.T_ideal#(- t_variation/space_size)
        return self.room.image,reward,True,{}

    def reset(self):
        self.room.heat_sources[1].T = np.random.randint(0, 10)
        self.room.heat_sources[0].T = np.random.choice([np.random.randint(15, 20),np.random.randint(35, 50)])
        self.room.image = self.room.heat_sources[0].T * np.random.rand(16, 16)
    def render(self, mode='human', close=False):
        heatmap = plt.imshow(self.room.image, cmap='hot', interpolation='nearest')
        plt.colorbar(heatmap)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        # fig.canvas.flush_ev

    def _get_obs(self):
        thermal_image = ...
        return thermal_image


    def _get_cost(self):

        self.mean((self.room.image-self.T_ideal)**2)
        return NotImplementedError