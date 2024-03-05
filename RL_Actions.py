# Configuration parameters for GM in the order: shift, moe, sync_ipd, upper bound
import subprocess, os, time

import numpy as np

from WiFi_Action_Simulation import Get_avg_reward_per_action
# File containing Metrics estimated by Bob
Metrics_file = 'Step_BLER'
actions_space = {0:('QPSK', 4, 10.0),
                 1:('QAM_16', 4, 10.0),
                 2:('QAM_64', 4, 10.0),
                 3:('QPSK', 8, 10.0),
                 4:('QAM_16', 8, 10.0),
                 5:('QAM_64', 8, 10.0),
                 6:('QPSK', 12, 10.0),
                 7:('QAM_16', 12, 10.0),
                 8:('QAM_64', 12, 10.0),

                 9: ('QPSK', 4, 15.0),
                 10: ('QAM_16', 4, 15.0),
                 11: ('QAM_64', 4, 15.0),
                 12: ('QPSK', 8, 15.0),
                 13: ('QAM_16', 8, 15.0),
                 14: ('QAM_64', 8, 15.0),
                 15: ('QPSK', 12, 15.0),
                 16: ('QAM_16', 12, 15.0),
                 17: ('QAM_64', 12, 15.0),

                 18: ('QPSK', 4, 20.0),
                 19: ('QAM_16', 4, 20.0),
                 20: ('QAM_64', 4, 20.0),
                 21: ('QPSK', 8, 20.0),
                 22: ('QAM_16', 8, 20.0),
                 23: ('QAM_64', 8, 20.0),
                 24: ('QPSK', 12, 20.0),
                 25: ('QAM_16', 12, 20.0),
                 26: ('QAM_64', 12, 20.0),

                 27: ('QPSK', 4, 25.0),
                 28: ('QAM_16', 4, 25.0),
                 29: ('QAM_64', 4, 25.0),
                 30: ('QPSK', 8, 25.0),
                 31: ('QAM_16', 8, 25.0),
                 32: ('QAM_64', 8, 25.0),
                 33: ('QPSK', 12, 25.0),
                 34: ('QAM_16', 12, 25.0),
                 35: ('QAM_64', 12, 25.0),

                 36: ('QPSK', 4, 30.0),
                 37: ('QAM_16', 4, 30.0),
                 38: ('QAM_64', 4, 30.0),
                 39: ('QPSK', 8, 30.0),
                 40: ('QAM_16', 8, 30.0),
                 41: ('QAM_64', 8, 30.0),
                 42: ('QPSK', 12, 30.0),
                 43: ('QAM_16', 12, 30.0),
                 44: ('QAM_64', 12, 30.0),


                 }

def agent_configure(action, avg_count):
    Modulation, num_bits, SNR = actions_space[action]
    n_samples , Similarity, BLER = Get_avg_reward_per_action(Modulation=Modulation, num_bits=num_bits, SNR=SNR, avg_count=avg_count)
    print('\n The returned BLER = {} and Simlarity = {} \n'.format(BLER, Similarity))

    # with open(Metrics_file) as f:
    #     BLER = int(f.readlines()[0])
    #
    if BLER + Similarity < 5:
        reward = 1
    elif BLER + Similarity < 30:
        reward = 0
    else:
        reward = -1
    return np.array([BLER, Similarity]), reward




