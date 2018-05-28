class HeatSource(object):
    def __init__(self,ix,iy,dx,dy,T):
        self.ix = ix
        self.iy = iy
        self.dx = dx
        self.dy = dy
        self.T = T

    def act_on_source(self,Tnew):
        self.T = Tnew

    def apply_source(self,img):
        img[self.ix:self.ix+self.dx:,self.iy:self.iy+self.dy:] = self.T
        return img

# class Action(object):
#     def __init__(self,heat_source):
#         self.source = heat_source


class HeatEnv(object):
    """
    Args:
        boundary: list of HeatSources
        source of changement of temperature
    """
    metadata = {}

    def __init__(self):
        self.boundary = []


    def step(self, a):
        reward = 0.0
        action = self._action_set[a]

        # if isinstance(self.frameskip, int):
        #     num_steps = self.frameskip
        # else:
        #     num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        # for _ in range(num_steps):
        #     reward += self.ale.act(action)
        ob = self._get_obs()
        reward = 0.0

        reward += -self._get_cost(ob)
        done = False
        return ob, reward, done, {}

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