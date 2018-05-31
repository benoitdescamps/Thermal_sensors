from heat_env import env
from Q_learning import define_Q
from matplotlib import pyplot as plt
import time
import numpy as np

import tensorflow as tf

ACTIONS = ['DO NOTHING','HEAT UP','COOL DOWN']
if __name__ == "__main__":

    Q_in, Q_out = define_Q()
    # create environment
    environment = env.HeatEnv(T_ideal=23)

    saver = tf.train.Saver()

    NROUNDS = 100
    with tf.Session() as sess:
        saver.restore(sess, "model/4500/thermal_model.ckpt")
        for i in range(NROUNDS):
            if (i%5)==0:
                print('Resetting Environment...')
                environment.reset()
            state = environment.room.image.reshape(1, 32, 32, 1).copy()
            Q_current = sess.run(Q_out, feed_dict={Q_in: state})

            print(Q_current)

            action = np.argmax(Q_current.ravel())

            T_old = np.mean(state)
            state, reward, _, _ = environment.step(action)
            T_new = np.mean(state)
            print(
                'Remaining rounds: {} \n Temparture old state:{:.2f} \n Temperature new state:{:.2f} \n Action Taken:{} \n Temperature Heater: {}'.format( \
                    NROUNDS - i, T_old, T_new, ACTIONS[action],
                    environment.room.heat_sources[0].T))

            plt.imshow(environment.room.image, cmap='hot', interpolation='nearest')
            plt.show()
            time.sleep(2)
