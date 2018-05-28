import numpy as np

def propagate(T_array,dx,dy,dt):
    return T_array[:-2:,:-2:] + ( np.diff(T_array,n=2,axis=0)+np.diff(T_array,n=2,axis=1) )*dt/(dx*dy)