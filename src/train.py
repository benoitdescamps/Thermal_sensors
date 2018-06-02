import tensorflow as tf
from heat_env import env

from Q_learning.utils import Replay_Memory_D
from Q_learning.net import define_Q,get_cost
import numpy as np
from matplotlib import pyplot as plt

import time

ACTIONS = ['DO NOTHING','HEAT UP','COOL DOWN']

def basic_Q(T,T_ideal,eps):
    return np.array([np.int(np.abs(T-T_ideal)<eps),\
                    -np.sign(T-T_ideal)*np.int(np.abs(T-T_ideal)>=eps),\
                    np.sign(T-T_ideal)*np.int(np.abs(T-T_ideal)>=eps)])

if __name__ == '__main__':
    plt.ion()

    ROOM_SHAPE = (16,16)

    Q_in,Q_out = define_Q(input_shape=ROOM_SHAPE)
    replay_memory_D = Replay_Memory_D(input_shape=ROOM_SHAPE,max_buffer_size=100,sample_size=50)

    #create environment
    T_ideal = 23
    environment = env.HeatEnv(room_shape=ROOM_SHAPE,T_ideal=T_ideal)
    q_p = 0.8
    q_eps = 1.
    epsilon = 0.2
    gamma = 0.1
    N_ROUNDS = 10000
    N_EPOCHS = 10
    N_TRAININ_ROUNDS = 10

    target = tf.placeholder(shape=(None,), dtype=tf.float32)
    action_indices = tf.placeholder(shape=(None,), dtype=tf.int32)
    cost = get_cost(target,Q_out,action_indices)

    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost)

    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "model/4000/thermal_model.ckpt")
        for round in range(N_ROUNDS):
            if (round % 50)==0:
                print('Resetting Environment...')
                environment.reset()

            old_state = environment.room.image.reshape( (1,)+ROOM_SHAPE+(1,) ).copy()

            environment.render()

            Told = environment.room.get_room_temperature()

            Q_current = sess.run(Q_out,feed_dict={Q_in:old_state})

            if np.random.rand()<epsilon:
                action = np.random.randint(1,3)
            else:
                action = np.argmax((1-q_p)*Q_current.ravel()+q_p*(basic_Q(Told,T_ideal,q_eps)))


            new_state,reward,_,_ = environment.step(action)
            Tnew = environment.room.get_room_temperature()
            if (round % 2)==0:
                print('Remaining rounds: {} \n  Temparture old state:{:.2f} \n  Temperature new state:{:.2f} \n  Action Taken:{} \n  Temperature Heater: {} \n  Temperature window: {} \n  Reward: {}'.format(\
                    N_ROUNDS-round,Told,Tnew, ACTIONS[action],environment.room.heat_sources[0].T,environment.room.heat_sources[1].T,reward))
            replay_memory_D.add(reward=reward,action=action,state=old_state,new_state=new_state)

            for epoch in range(N_EPOCHS):
                sampled_targets, sampled_states, sampled_actions = replay_memory_D.sample(sess, Q_in, Q_out, gamma)
                for n in range(N_TRAININ_ROUNDS):
                    sess.run(optimizer,feed_dict={Q_in:sampled_states,action_indices:sampled_actions,target:sampled_targets})
            # Save the variables to disk.

            if (round % 500) == 0:
                save_path = saver.save(sess, "model/{}/thermal_model.ckpt".format(round))


