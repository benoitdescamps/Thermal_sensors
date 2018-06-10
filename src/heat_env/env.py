import numpy as np
from  .Room import Room, HeatSource
from matplotlib import pyplot as plt
from .server import Client
class HeatEnv(object):
    """
    Class for the Reinforcement Learning Environment
    Args:
        Tuple room_shape: dimension of the room
        Int T_ideal: Temperature which we would like to achieve
        Int TIME_STEPS: [Development] number of time steps which we should apply
        at each iteration of the environment
    """
    metadata = {}

    def __init__(self,room_shape=(16,16),T_ideal=23,TIME_STEPS=10):
        plt.ion()

        self.TIME_STEPS = TIME_STEPS
        self.T_ideal = T_ideal

        self.room = Room(image=15*np.random.rand(room_shape[0],room_shape[1]))
        hsrc0 = HeatSource(T=20,x0=4,x1=12,y0=5,y1=7,name='radiator',controllable=True)
        hsrc1 = HeatSource(T=10, x0=6, x1=10, y0=1, y1=2, name='window',controllable=False)
        self.room.add_heat_source(hsrc0)
        self.room.add_heat_source(hsrc1)

        self.client = Client()

        self.info = {'heatloss':list(),'T_room':list()}
        self.latest_info = {'heatloss':None,'T_room':None}


    def step(self, a):
        """
        Single iteration of the environment given an action a
        :param Int a: action
        :return:
        """
        if a ==1:
            self.room.heat_sources[0].T += 1.
            if self.room.heat_sources[0].T>50:
                self.room.heat_sources[0].T = 50
        elif a==2:
            self.room.heat_sources[0].T += -1.
            if self.room.heat_sources[0].T<10:
                self.room.heat_sources[0].T = 10

        heatloss = self.room.propagate(dt=0.2,dx=1.,dy=1.,n_steps=self.TIME_STEPS)

        assert(len(self.room.image.shape)==2)
        T_room = self.room.get_room_temperature()#np.sign(T_room-self.T_ideal), +(self.room.heat_sources[0].T-self.T_ideal >0)*heatloss
        reward = -np.sign(heatloss)*np.exp(-np.abs(heatloss))#np.exp(-np.abs(np.median(self.room.image)-self.T_ideal)/0.5 )

        self.latest_info['heatloss'] = heatloss
        self.latest_info['T_room'] = T_room
        self.info['heatloss'].append(heatloss)
        self.info['T_room'].append(T_room)
        return self.room.image,reward,True,{}

    def reset(self):
        """
        Resets the environment
        :return:
        """
        self.room.heat_sources[1].T = np.random.randint(0, 10)
        self.room.heat_sources[0].T = np.random.choice([np.random.randint(15, 20),np.random.randint(35, 50)])
        self.room.image = self.room.heat_sources[0].T * np.random.rand(16, 16)

        self.info = {'heatloss': list(),'T_room':list()}
        self.latest_info = {'heatloss': None, 'T_room': None}
    def render(self, mode='human', close=False):
        """
        renders the environment.
        :param mode:
        :param close:
        :return:
        """
        #TODO: fix server
        #TODO: connect data stream with Bokeh

        #f, (ax2, ax3) = plt.subplots(2, sharex=True, sharey=False)
        #heatmap = ax1.imshow(self.room.image, cmap='hot', interpolation='nearest')
        #plt.colorbar(heatmap,ax=[ax1])
        #ax2.plot(-np.cumsum(self.info['heatloss']))
        #ax3.plot(self.info['T_room'])
        #f.subplots_adjust(hspace=0)
        #f.canvas.draw_idle()
        #plt.draw()
        #plt.pause(0.0001)
        #plt.clf()
        # fig.canvas.flush_ev

        self.client.send_data(self.latest_info)

    def _get_obs(self):
        #TODO: later once I connect the sensors
        thermal_image = ...
        return thermal_image
