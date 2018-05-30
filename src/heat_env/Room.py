import numpy as np

class HeatSource(object):
    def __init__(self,T,x0,x1,y0,y1,name):
        self.name = name
        self.T = T
        self.x0 = x0
        self.x1 =x1
        self.y0 = y0
        self.y1 =y1

    def act_on_source(self,Tnew):
        self.T = Tnew

    def apply_source(self,image):
        image[self.x0:self.x1,self.y0:self.y1] = self.T
        return image
class Room(object):

    def __init__(self,image):
        self.image = image
        self.heat_sources = list()

    def add_heat_source(self,heatsrc):
        self.heat_sources.append(heatsrc)

    def _apply_heat_sources(self):
        for heatsrc in self.heat_sources:
            heatsrc.apply_source(self.image)

    def propagate(self,dt,dx,dy,n_steps):
        'appl'
        self.image[1:-1:,1:-1:] = self.image[1:-1:, 1:-1:] + dt*(np.diff(self.image, n=2, axis=0)[:,1:-1:]/(dx*dx) \
                                               + np.diff(self.image, n=2, axis=1)[1:-1:,:]/(dy*dy))


        return self._apply_heat_sources()

