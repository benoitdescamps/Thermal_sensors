import numpy as np
def basic_Q(T,T_ideal,eps):
    return np.array([np.int(np.abs(T-T_ideal)<eps),\
                    -np.sign(T-T_ideal)*np.exp(-np.abs(T-T_ideal)),\
                    np.sign(T-T_ideal)*np.exp(-np.abs(T-T_ideal))])

def _get_state(T,T_ideal):
    return int(T-T_ideal >0)