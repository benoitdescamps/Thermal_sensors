import numpy as np
class Replay_Memory_D(object):
    """
    Buffer class implementing the experience replay

    Args:
        Tuple input_shape: dimension fo the thermal image
        Int max_buffer_size: maximum size of processed event being buffered
        Int sample_size: size of events sampled for training
    """
    def __init__(self,input_shape=(16,16),max_buffer_size=100,sample_size=10):
        self.input_shape = input_shape
        self.max_buffer_size = max_buffer_size
        self.sample_size = sample_size

        self.reward_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()
        self.new_state_buffer = list()

        self.current_size = 0
    def add(self,reward: float,action:float ,state: np.array,new_state: np.array):
        """
        Add new event to the before for later use in the experience replay
        :param float reward:
        :param Int action:
        :param np.array state:
        :param np.array new_state:
        :return:
        """
        self.current_size +=1
        self.reward_buffer += [reward]
        self.action_buffer += [action]
        self.state_buffer += [state]
        self.new_state_buffer += [new_state]

        if (self.current_size>self.max_buffer_size):
            self.current_size = self.max_buffer_size
            self.reward_buffer = self.reward_buffer[-self.max_buffer_size::]
            self.action_buffer = self.action_buffer[-self.max_buffer_size::]
            self.state_buffer = self.state_buffer[-self.max_buffer_size::]
            self.new_state_buffer = self.new_state_buffer[-self.max_buffer_size::]

    def sample(self,sess,Q_in,Q_out,gamma):
        """
        Samples the events from the buffer

        Args:

        :param tf.Session sess: tensorflow session
        :param tf.tensor Q_in: input tensor of the current Q-matrix at the time of sampling
        :param tf.tensor Q_out: output tensor of the current Q-matrix at the time of sampling
        :param float gamma: reinforcement learning parameter. Weights the current reward versus future rewards.
        :return: Tuple[np.array, np.array, np.array] (target samples, states sample, actions sample)
        """
        indices = list(np.random.choice(self.current_size, np.min([self.current_size,self.sample_size])) )

        sampled_targets = np.array([self.reward_buffer[i] + gamma * np.max(
            sess.run(Q_out, feed_dict={Q_in: self.new_state_buffer[i].reshape(  (1,)+self.input_shape+(1,) )}).ravel())\
                        for i in indices])
        sampled_states = np.concatenate([self.state_buffer[i].reshape( (1,)+self.input_shape+(1,) ) for i in indices],axis=0)
        sampled_actions = np.array([self.action_buffer[i] for i in indices])

        return sampled_targets,sampled_states,sampled_actions


