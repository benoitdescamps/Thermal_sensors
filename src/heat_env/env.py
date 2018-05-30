
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

    def __init__(self):
        self.room = Room(image=20*np.ones(shape=(32,32)))
        hsrc0 = HeatSource(T=23,x0=4,x1=20,y0=4,y1=7,name='radiator')
        hsrc1 = HeatSource(T=10, x0=4, x1=28, y0=0, y1=1, name='window')
        self.room.add_heat_source(hsrc0)
        self.room.add_heat_source(hsrc1)


    def step(self, a):
        reward = 0.0
        #action = self._action_set[a]

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
        self.room.propagate(dt=1.,dx=0.1,dy=0.1,n_steps=100)
        return None,None,True,{}

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def _get_obs(self):
        thermal_image = ...
        return thermal_image

    def _propagate(self, img, dt, dx, dy, n_steps):
        """
        Apply the heat equations

        Args:
            img: numpy array:
            thermal image
            dt: float:
            time delta-ste[
            dx: float:
            x-space delta-step
            dy: float:
            y-space delta-step


        """
        new_img = img.copy()
        new_img[:-2:, :-2:] = img[:-2:, :-2:] + (np.diff(img, n=2, axis=0) + np.diff(img, n=2, axis=1)) * dt / (dx * dy)

        return self._apply_boundary_conditions(
            new_img
        )

    def _apply_boundary_conditions(self, img):
        """
        Apply boundary conditions
        :param img:
        :return:
        """
        return NotImplementedError

    def _get_cost(self,img):
        return NotImplementedError