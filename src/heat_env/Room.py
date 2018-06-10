import numpy as np

class HeatSource(object):
    def __init__(self,T,x0,x1,y0,y1,name,controllable=False):
        self.name = name
        self.T = T
        #self.Told = T
        self.x0 = x0
        self.x1 =x1
        self.y0 = y0
        self.y1 =y1
        self.is_oontrollable = controllable
    def act_on_source(self,Tnew):
        self.T = Tnew

    def apply_source(self,image):
        image[self.x0:self.x1,self.y0:self.y1] = self.T
        self.Told = self.T
        return image
    def get_heat_loss(self,img):
        return np.mean(img[self.x0:self.x1,self.y0:self.y1]-self.T)
    def get_heat_img(self,img):
        return img[self.x0:self.x1,self.y0:self.y1]
class Room(object):

    def __init__(self,image):
        self.image = image
        self.heat_sources = list()

    def add_heat_source(self,heatsrc):
        self.heat_sources.append(heatsrc)

    def _apply_heat_sources(self):
        heat_loss = 0.0
        for heatsrc in self.heat_sources:
            if heatsrc.is_oontrollable:
                heat_loss+= heatsrc.get_heat_loss(self.image)
            heatsrc.apply_source(self.image)
        return heat_loss

    def propagate(self,dt,dx,dy,n_steps):
        'appl'
        heat_loss = 0.0
        for n in range(n_steps):
            self.image[1:-1:,1:-1:] = self.image[1:-1:, 1:-1:] + dt*(np.diff(self.image, n=2, axis=0)[:,1:-1:]/(dx*dx) \
                                               + np.diff(self.image, n=2, axis=1)[1:-1:,:]/(dy*dy))
            self.image[0, :] = self.image[1, :]
            self.image[:, 0] = self.image[:, 1]
            self.image[-1, :] = self.image[-2, :]
            self.image[:, -1] = self.image[:, -2]

            heat_loss += self._apply_heat_sources()*dx*dy*dt
        return heat_loss

    def get_room_temperature(self):
        return np.median(self.image)



