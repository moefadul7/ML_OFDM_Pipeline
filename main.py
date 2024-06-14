########################################################################################################################
########### This script is the main controller for DSSS coding/decoding as well as OFDM PHY Tx and Rx ##################
########################################################################################################################
############################# Author: Mohamed Fadul , Email: Mohammed-fadul@utc.edu ####################################
########################################################################################################################
####### Package Includes trained instances of Alice & Bob. You can switch the training on for  fresh models ############
########################################################################################################################
import sys, os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import subprocess
import time

from DL_DSSS.Alice import Alice_Load_Quantize
from DL_DSSS.Bob import Bob_Load_Dequantize
from DL_DSSS.Bob import Bob_Channel_Selection
from DSSS_Train import NN_setup, framework_train
# from Modulation.Modulation import QAM_64, QAM_16, QPSK, OFDM, BER
# from Modulation.Channel import Channel, Apply_Channel
# from Transmit_PHY_OFDM import Send_OFDM
# from Receive_PHY_OFDM import Recv_OFDM
#from bit_loader import Dump_Rcvd_stream
from DL_DSSS.Loss import Loss
from DL_DSSS import Models

########################################################################################################################
#tf.keras.utils.get_custom_objects().update({'OneHotLayer': Models.OneHotLayer})
# Parameter Setup:

# Set up the DSSS parameters: message, code, and spread signal
m_bits = 16
k_bits = 15
c_bits = 16

num_bits = 16

pad = 'same'

# Compute the size of the message space, used later in training
m_train = 2**(m_bits)

# Training parameters:
n_epoch = 100
batch_size = 1024
n_batches = m_train // batch_size
abecycles = 1
evecycles = 2
n_samples = 1

# Switch Training on to train a new instance or otherwise use the saved model
Train = False



# File containing the saved model
model_file = 'Alice_Bob_Model_Rogers.keras'


"""
# OFDM parameters:
K = 64  # number of sub-carriers
CP = 16  # length of the cyclic prefix
P = 8  # number of pilot carriers
pilotValue = 3+3j #pilot carrier value
#SNR = 20  # DB

With_Multipath = True
"""
# File including processed bit_stream at Tx by BYU
# replace this with the processed file to be transmitted
file_name_A = 'Alice_output_stream.bin'

# File including processed bit_stream at Rx by BYU
# replace this with the processed file to be decoded by Bob
file_name_B = 'Bob_input_stream.bin'

# Configurable variables that construct the action-vector in order:
"""
if (len(sys.argv) != 4):
    print(""You need to provide 3 configurable parameters including \n Modulation per Carrier \n Q bits \n SNR"")
    exit()

Mod_Per_Carrier = sys.argv[1]

# Number of Quantization bits
num_bits = int(sys.argv[2])

SNR = float(sys.argv[3])

# Function to define the modulation type
def Mapping(scheme, D):
    return eval(scheme + '(D)')
"""



########################################################################################################################

if Train:
    # Setup Neural Networks for Alice, Bob, Eve
    Alice_Model, Bob_Model, Eve_Model, abmodel, evemodel = NN_setup(m_bits, k_bits, c_bits)

    # Joint Training of Alice, Bob, & Eve
    framework_train(m_bits, k_bits, n_batches, batch_size, n_epoch, Alice_Model, Bob_Model, abmodel, evemodel,
                        abecycles, evecycles, n_samples, model_file)


frequencies = [5170000000.0, 5180000000.0, 5190000000.0, 5200000000.0, 5210000000.0, 5220000000.0, 5230000000.0,
               5240000000.0, 5250000000.0, 5260000000.0, 5270000000.0, 5280000000.0, 5290000000.0, 5300000000.0,
               5310000000.0, 5320000000.0, 5500000000.0, 5510000000.0, 5520000000.0, 5530000000.0, 5540000000.0,
               5550000000.0, 5560000000.0, 5570000000.0, 5580000000.0, 5590000000.0, 5600000000.0, 5610000000.0,
               5620000000.0, 5630000000.0, 5640000000.0, 5660000000.0, 5670000000.0, 5680000000.0, 5690000000.0,
               5700000000.0, 5710000000.0, 5720000000.0, 5745000000.0, 5755000000.0, 5765000000.0, 5775000000.0,
               5785000000.0, 5795000000.0, 5805000000.0, 5825000000.0, 5860000000.0, 5870000000.0, 5880000000.0,
               5890000000.0, 5900000000.0, 5910000000.0, 5920000000.0]


Alice_out_quantized, Alice_out_stream, Alice_out, Alice_channel, tst_msg, tst_key = Alice_Load_Quantize(model_file, n_samples, num_bits)


def softmax_to_one_hot(softmax_vector):
    one_hot = np.zeros_like(softmax_vector)
    one_hot[np.arange(len(softmax_vector)), np.argmax(softmax_vector, axis=1)] = 1
    return one_hot

# print("Alice_Channel (softmax)", Alice_channel)

Alice_channel_one_hot = softmax_to_one_hot(Alice_channel)
# print("Alice_Channel (one_hot)", Alice_channel_one_hot)

# print("Alice_out", Alice_out)
A_chan_index = np.argmax(Alice_channel_one_hot, axis=1)
print("A_chan_index",A_chan_index)
freq_tx = frequencies[A_chan_index[0]]
print("freq_tx",freq_tx)

"""
wifi_tx_process = subprocess.Popen(['python3', 'wifi_tx.py', '--freq_tx', str(freq_tx)])
time.sleep(30)
wifi_tx_process.terminate()
wifi_tx_process.wait()
"""
Bob_channel = Bob_Channel_Selection(model_file, tst_key)
# print("Bob_channel (softmax)", Bob_channel)

Bob_channel_one_hot = softmax_to_one_hot(Bob_channel)
# print("Bob_channel (one_hot)", Bob_channel_one_hot)

B_chan_index = np.argmax(Bob_channel_one_hot, axis=1)
print("B_chan_index",B_chan_index)
freq_rx = frequencies[B_chan_index[0]]
print("freq_rx", freq_rx)
"""
subprocess.run(['python3', 'wifi_rx.py', str(freq_rx)])

## Ghost Modulation Here

# Create OFDM Object
ofdm = OFDM(K, P, CP, pilotValue)

# Set up a modulation object 'Mapper'
Mapper = Mapping(Mod_Per_Carrier, len(ofdm.data_carriers))

# Transmit OFDM symbols with payloads from 'file_name'
Tx_bit_stream, OFDM_CP = Send_OFDM(file_name_A, Mapper, ofdm)

# Apply wireless channel
print('\n Applying Wireless Channel Response ...')
channel = Channel(ofdm.K, SNR)
Received_Signals = Apply_Channel(OFDM_CP, channel, With_Multipath)

# Receive and process OFDM symbols
Rcv_bit_stream = Recv_OFDM(Received_Signals, ofdm, Mapper)


# BER calculation
BER, span = BER(Tx_bit_stream, Rcv_bit_stream)
print("\n BER for a transmitted bit stream of length %d is %.8f" % (len(Tx_bit_stream), BER))

# save received bit stream
Dump_Rcvd_stream(file_name_B, Rcv_bit_stream, span)

## Ghost Demodulation Here ...

# Extract and Dequantize DSSS encoded message at the receiver
#Bob_in = Load_Bob_input(file_name_B, c_bits, num_bits, Mapper)
Bob_in, Bob_out, Bob_chan = Bob_Load_Dequantize(model_file, file_name_B, c_bits, num_bits, tst_key)

# assert np.array_equal(Alice_channel, Bob_channel), "Channel Synchronization Failed!"
Total_Loss = Loss(tst_msg, Bob_out)
Error_Per_Msg = np.where(np.sum(abs(tst_msg-Bob_out), axis = 1) > 0, 1, 0)
BLER = np.sum(Error_Per_Msg)
print('\n The total End-to-End BLER over {} messages is {:.6f}'.format(n_samples, BLER))
"""