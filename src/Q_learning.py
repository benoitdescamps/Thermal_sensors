import tensorflow as tf
from heat_env import env
import numpy as np
from matplotlib import pyplot as plt

class Replay_Memory_D(object):
    def __init__(self,max_buffer_size=100,sample_size=10):
        self.max_buffer_size = max_buffer_size
        self.sample_size = sample_size

        self.reward_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()
        self.new_state_buffer = list()

        self.current_size = 0
    def add(self,reward,action,state,new_state):
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
        indices = list(np.random.choice(self.current_size, np.min([self.current_size,self.sample_size])) )

        sampled_targets = np.array([self.reward_buffer[i] + gamma * np.max(
            sess.run(Q_out, feed_dict={Q_in: self.new_state_buffer[i].reshape(1, 32, 32, 1)}).ravel())\
                        for i in indices])
        sampled_states = np.concatenate([self.state_buffer[i].reshape(1,32,32,1) for i in indices],axis=0)
        sampled_actions = np.array([self.action_buffer[i] for i in indices])

        return sampled_targets,sampled_states,sampled_actions


def define_Q():
    input = tf.placeholder(shape=(None, 32, 32, 1), dtype=tf.float32)
    nn_1 = tf.layers.batch_normalization(input)
    filter_1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=1.0))
    layer_1 = tf.nn.conv2d(input=nn_1, strides=[1, 1, 1, 1], filter=filter_1, padding='VALID')
    filter_2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=1.0))
    layer_2 = tf.nn.conv2d(input=layer_1, strides=[1, 1, 1, 1], filter=filter_2, padding='VALID')
    filter_3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=1.0))
    layer_3 = tf.nn.conv2d(input=layer_2, strides=[1, 1, 1, 1], filter=filter_3, padding='VALID')
    #layer_2 = tf.layers.max_pooling2d(inputs=layer_1, pool_size=(4, 4), strides=(1, 1))
    #remove maxpooling layer as translational invariance might not be necessary here
    layer_3 = tf.layers.dense(inputs=tf.layers.Flatten()(layer_3) \
                              , units=8, activation=tf.nn.relu, use_bias=True)
    output = tf.layers.dense(inputs=layer_3, units=3, use_bias=True)
    return input,output

def get_cost(target,Q,action_indices):
    row_indices =  tf.range(tf.shape(action_indices)[0])
    full_indices = tf.stack([row_indices, action_indices], axis=1)
    q_values = tf.gather_nd(Q, full_indices)


    return tf.losses.mean_squared_error(labels=target,predictions=q_values)

ACTIONS = ['DO NOTHING','HEAT UP','COOL DOWN']
if __name__ == '__main__':
    Q_in,Q_out = define_Q()
    #replay_memory_D = list()
    replay_memory_D = Replay_Memory_D(max_buffer_size=100,sample_size=50)
    #{'state':None,'action':None,'reward':None,'new_state':None}


    #create environment
    environment = env.HeatEnv(T_ideal=23)
    epsilon = 0.2
    gamma = 0.1
    N_ROUNDS = 5000
    N_EPOCHS = 10
    N_TRAININ_ROUNDS = 10

    target = tf.placeholder(shape=(None,), dtype=tf.float32)
    action_indices = tf.placeholder(shape=(None,), dtype=tf.int32)
    cost = get_cost(target,Q_out,action_indices)

    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "model/500/thermal_model.ckpt")
        for round in range(N_ROUNDS):
            if (round % 50)==0:
                print('Resetting Environment...')
                environment.reset()

            old_state = environment.room.image.reshape(1,32,32,1).copy()
            Q_current = sess.run(Q_out,feed_dict={Q_in:old_state})

            if np.random.rand()<epsilon:
                action = np.random.randint(0,3)
            else:
                action = np.argmax(Q_current.ravel())


            new_state,reward,_,_ = environment.step(action)

            if (round % 2)==0:
                print('Remaining rounds: {} \n  Temparture old state:{:.2f} \n  Temperature new state:{:.2f} \n  Action Taken:{} \n  Temperature Heater: {} \n  Temperature window: {} \n  Reward: {}'.format(\
                    N_ROUNDS-round,np.mean(old_state[:,10:,10:,:]), np.mean(new_state[5:,5:]), ACTIONS[action],environment.room.heat_sources[0].T,environment.room.heat_sources[1].T,reward))
            replay_memory_D.add(reward=reward,action=action,state=old_state,new_state=new_state)

            for epoch in range(N_EPOCHS):
                sampled_targets, sampled_states, sampled_actions = replay_memory_D.sample(sess, Q_in, Q_out, gamma)
                for n in range(N_TRAININ_ROUNDS):
                    sess.run(optimizer,feed_dict={Q_in:sampled_states,action_indices:sampled_actions,target:sampled_targets})
            # Save the variables to disk.

            if (round % 500) == 0:
                save_path = saver.save(sess, "model/{}/thermal_model.ckpt".format(round))


