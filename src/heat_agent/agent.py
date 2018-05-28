import numpy as np

class Heat_Agent(object):
    """
    Agent Class interacting with the thermal image.
    Each action of the agent affects the  Heat Equation.

    Args:

    """
    def __init__(selfs):
        pass

    def _propagate(self,img,dt,dx,dy,n_steps):
        'appl'
        new_img = img.copy()
        new_img[1:-1:,1:-1:] = img[1:-1:, 1:-1:] + dt*(np.diff(img, n=2, axis=0)[:,1:-1:]/(dx*dx) \
                                               + np.diff(img, n=2, axis=1)[1:-1:,:]/(dy*dy))


        return self._apply_boundary_conditions(
            new_img
        )

    def _apply_boundary_conditions(self,img):
        return NotImplementedError

    def actions(self):
        return NotImplementedError