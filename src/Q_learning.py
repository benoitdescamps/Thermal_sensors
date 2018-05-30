import tensorflow as tf
from heat_env import env
import numpy as np
from matplotlib import pyplot as plt


def define_Q():
    input = tf.placeholder(shape=(None, 32, 32, 1), dtype=tf.float32)
    filter = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.1))
    layer_1 = tf.nn.conv2d(input=input, strides=[1, 1, 1, 1], filter=filter, padding='SAME')
    layer_2 = tf.layers.max_pooling2d(inputs=layer_1, pool_size=(4, 4), strides=(1, 1))
    #remove maxpooling layer as translational invariance might not be necessary here
    layer_3 = tf.layers.dense(inputs=tf.layers.Flatten()(layer_2) \
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
    replay_memory_D = list()
    #{'state':None,'action':None,'reward':None,'new_state':None}


    #create environment
    environment = env.HeatEnv()
    epsilon = 0.2
    gamma = 0.1
    N_ROUNDS = 10000
    N_TRAININ_ROUNDS = 100

    target = tf.placeholder(shape=(None,), dtype=tf.float32)
    action_indices = tf.placeholder(shape=(None,), dtype=tf.int32)
    cost = get_cost(target,Q_out,action_indices)

    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for round in range(N_ROUNDS):
            if (round % 100)==0:
                environment.reset()

            old_state = environment.room.image.reshape(1,32,32,1).copy()
            Q_current = sess.run(Q_out,feed_dict={Q_in:old_state})

            if np.random.rand()<epsilon:
                action = np.random.randint(0,3)
            else:
                action = np.argmax(Q_current.ravel())


            new_state,reward,_,_ = environment.step(action)

            if (round % 500)==0:
                print('Temparture old state:{:.2f} \n Temperature new state:{:.2f} \n Action Taken:{}'.format(
                    np.mean(old_state), np.mean(new_state), ACTIONS[action]))

            replay_memory_D.append({'state':old_state,\
                    'action':action,'reward':reward,'new_state':new_state})
            if len(replay_memory_D)>10:
                replay_memory_D = replay_memory_D[-10::]


            tt = list()
            aa = list()
            ss = list()
            cost =  0.0

            for event in replay_memory_D:
                tt +=[event['reward']+gamma*np.max(sess.run(Q_out,feed_dict={Q_in:event['new_state'].reshape(1,32,32,1)}).ravel())]
                aa += [event['action']]
                ss += [(event['state'].reshape(1,32,32,1))]

            tt = np.array(tt)
            ss = np.concatenate(ss,axis=0)
            aa = np.array(aa).ravel()



            for n in range(N_TRAININ_ROUNDS):
                sess.run(optimizer,feed_dict={Q_in:ss,action_indices:aa,target:tt})


