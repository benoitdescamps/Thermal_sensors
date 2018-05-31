
# class Action(object):
#     def __init__(self,heat_source):
#         self.source = heat_source

import numpy as np
from  .Room import Room, HeatSource
class HeatEnv(object):
    """
    Args:
        boundary: list of HeatSources
        source of changement of temperature
    """
    metadata = {}

    def __init__(self,T_ideal=23):
        self.T_ideal = T_ideal

        self.room = Room(image=15*np.random.rand(32,32))
        hsrc0 = HeatSource(T=20,x0=4,x1=28,y0=2,y1=4,name='radiator')
        hsrc1 = HeatSource(T=10, x0=8, x1=14, y0=0, y1=1, name='window')
        self.room.add_heat_source(hsrc0)
        self.room.add_heat_source(hsrc1)


    def step(self, a):
        reward = 0.0
        #action = self._action_set[a]

        if a ==1:
            self.room.heat_sources[0].T += 5.
            if self.room.heat_sources[0].T>200:
                self.room.heat_sources[0].T = 200
        elif a==2:
            self.room.heat_sources[0].T += -5.
            if self.room.heat_sources[0].T<10:
                self.room.heat_sources[0].T = 10

        #t_variation = np.sum( np.abs(self.room.image-self.T_ideal) )
        #space_size = self.room.image.size
        #for heatsrc in self.room.heat_sources:
        #    t_variation+= - np.sum(np.abs(heatsrc.get_heat_img(self.room.image)-self.T_ideal))
        #    space_size+= - heatsrc.get_heat_img(self.room.image).size

        heat_loss = self.room.propagate(dt=0.2,dx=1.,dy=1.,n_steps=1000)
        #*np.abs(self.T_ideal-self.room.heat_sources[0].T)
        assert(len(self.room.image.shape)==2)
        reward = np.mean(np.abs(self.room.image[10::,10::]-self.T_ideal))#(- t_variation/space_size)
        return self.room.image,reward,True,{}

    def reset(self):
        self.room.heat_sources[1].T = np.random.randint(0, 10)
        self.room.heat_sources[0].T = np.random.randint(15, 20)
        self.room.image = self.room.heat_sources[0].T * np.random.rand(32, 32)
    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        thermal_image = ...
        return thermal_image


    def _get_cost(self):

        self.mean((self.room.image-self.T_ideal)**2)
        return NotImplementedError