from heat_env import env
from matplotlib import pyplot as plt
import time
import numpy as np

if __name__ == "__main__":
    environment = env.HeatEnv()

    for i in range(10):
        _, heat_loss, _, _ = environment.step(a=None)

        plt.imshow(environment.room.image, cmap='hot', interpolation='nearest')

        print(np.mean(environment.room.image),heat_loss)
        plt.show()
        time.sleep(2)
