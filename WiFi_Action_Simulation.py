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
import matplotlib.pyplot as plt
from DL_DSSS.Alice import Alice_Load_Quantize
from DL_DSSS.Bob import Bob_Load_Dequantize
from DSSS_Train import NN_setup, framework_train
from Modulation.Modulation import QAM_64, QAM_16, QPSK, OFDM, BER
from Modulation.Channel import Channel, Apply_Channel
from Transmit_PHY_OFDM import Send_OFDM
from Receive_PHY_OFDM import Recv_OFDM
from bit_loader import Dump_Rcvd_stream
from DL_DSSS.Loss import Loss

########################################################################################################################

def Get_avg_reward_per_action(**Kwargs):

    # Parameter Setup:
    # Set up the DSSS parameters: message, code, and spread signal
    m_bits = 16
    k_bits = 15
    c_bits = 16
    pad = 'same'

    # Compute the size of the message space, used later in training
    m_train = 2**(m_bits)

    # Training parameters:
    n_epoch = 1
    batch_size = 512
    n_batches = m_train // batch_size
    abecycles = 1
    evecycles = 2


    # Switch Training on to train a new instance or otherwise use the saved model
    Train = False

    # File containing the saved model
    model_file = 'Alice_Bob_Model.keras'

    # OFDM parameters:
    K = 64  # number of sub-carriers
    CP = 16  # length of the cyclic prefix
    P = 8  # number of pilot carriers
    pilotValue = 3+3j #pilot carrier value
    #SNR = 20  # DB

    With_Multipath = True

    # File including processed bit_stream at Tx by BYU
    # replace this with the processed file to be transmitted
    file_name_A = 'Alice_output_stream'

    # File including processed bit_stream at Rx by BYU
    # replace this with the processed file to be decoded by Bob
    file_name_B = 'Bob_input_stream'

    # Configurable variables that construct the action-vector in order:

    if (len(Kwargs) != 4):
        print("""Exception! You need to provide 4 configurable parameters including \n Modulation per Carrier \n Q bits \n SNR \n num_samples""")
        exit()

    Mod_Per_Carrier = Kwargs['Modulation']

    # Number of Quantization bits
    num_bits = Kwargs['num_bits']

    SNR = Kwargs['SNR']

    n_samples = Kwargs['avg_count']


    # Function to define the modulation type
    def Mapping(scheme, D):
        return eval(scheme + '(D)')




    ########################################################################################################################

    if Train:
        # Setup Neural Networks for Alice, Bob, Eve
        Alice_Model, Bob_Model, Eve_Model, abmodel, evemodel = NN_setup(m_bits, k_bits, c_bits)

        # Joint Training of Alice, Bob, & Eve
        framework_train(m_bits, k_bits, n_batches, batch_size, n_epoch, Alice_Model, Bob_Model, abmodel, evemodel,
                            abecycles, evecycles, n_samples, model_file)


    Alice_out_quantized, Alice_out_stream, Alice_out, tst_msg, tst_key = Alice_Load_Quantize(model_file, n_samples, num_bits)

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
    Ber, span = BER(Tx_bit_stream, Rcv_bit_stream)
    print("\n BER for a transmitted bit stream of length %d is %.8f" % (len(Tx_bit_stream), Ber))

    # save received bit stream
    Dump_Rcvd_stream(file_name_B, Rcv_bit_stream, span)

    ## Ghost Demodulation Here ...

    # Extract and Dequantize DSSS encoded message at the receiver
    #Bob_in = Load_Bob_input(file_name_B, c_bits, num_bits, Mapper)
    Bob_in, Bob_out = Bob_Load_Dequantize(model_file, file_name_B, c_bits, num_bits, Mapper, tst_key)


    Total_Loss = Loss(tst_msg, Bob_out)
    Similarity_Per_Msg = np.sum(abs(tst_msg - Bob_out), axis=1)
    BLER_Per_Msg = np.where(Similarity_Per_Msg > 0, 1, 0)
    #BLER = np.sum(Error_Per_Msg)

    return n_samples, np.mean(Similarity_Per_Msg), np.sum(BLER_Per_Msg)

if __name__ == '__main__':
    n_samples, Similarity, BLER = Get_avg_reward_per_action(Modulation='QAM_64', num_bits=8, SNR=30.0, avg_count=100)
    print('\n The total End-to-End Simlarity and BLER over {} messages is {:.6f} and {:.6f}'.format(n_samples, Similarity, BLER))


