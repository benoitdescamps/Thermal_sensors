
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

        self.room = Room(image=20*np.ones(shape=(32,32)))
        hsrc0 = HeatSource(T=35,x0=4,x1=20,y0=2,y1=3,name='radiator')
        hsrc1 = HeatSource(T=10, x0=4, x1=28, y0=0, y1=1, name='window')
        self.room.add_heat_source(hsrc0)
        self.room.add_heat_source(hsrc1)


    def step(self, a):
        reward = 0.0
        #action = self._action_set[a]
        if a ==1:
            self.room.heat_sources[0].T += 5.
            if self.room.heat_sources[0].T>35:
                self.room.heat_sources[0].T = 35
        elif a==2:
            self.room.heat_sources[0].T += -5.
            if self.room.heat_sources[0].T<10:
                self.room.heat_sources[0].T = 10
        # if isinstance(self.frameskip, int):
        #     num_steps = self.frameskip
        # else:
        #     num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        # for _ in range(num_steps):
        #     reward += self.ale.act(action)

        #ob = self._get_obs()
        #reward = 0.0

        #reward += -self._get_cost(ob)
        #done = False
        #return ob, reward, done, {}
        heat_loss = self.room.propagate(dt=0.2,dx=1.,dy=1.,n_steps=1000)

        reward = -0.5*heat_loss - np.mean(np.abs(self.room.image-self.T_ideal))-0.2*np.abs(self.T_ideal-self.room.heat_sources[0].T)
        return self.room.image,reward,True,{}

    def reset(self):
        self.room.heat_sources[1].T = np.random.randint(10, 15)

    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        thermal_image = ...
        return thermal_image


    def _get_cost(self):

        self.mean((self.room.image-self.T_ideal)**2)
        return NotImplementedError