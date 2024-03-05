# This script is used to process received OFDM symbols

import numpy as np
import matplotlib.pyplot as plt
from bit_loader import bit_stream_loader
from Modulation.Modulation import  QAM_16, Serial_to_Parallel, Parallel_to_Serial, Bits_to_Symbols, Demodulation
from Modulation.Modulation import OFDM, OFDM_symbol, OFDM_time, WithCP, WithoutCP, OFDM_freq, Extract_payloads, BER
from Modulation.Channel import Channel, Apply_Channel, Channel_Est, Channel_Rem

def Recv_OFDM(Received_Signals, ofdm, Mapper):
    print("\n Collecting OFDM symbols by the receiver ...")
    # Remove CP
    Received_No_CP = WithoutCP(Received_Signals, ofdm)
    #print("\n Number of samples in the received signals without CP: ", Received_No_CP.shape)

    # FFT - Time domain
    Received_OFDM_Freq = OFDM_freq(Received_No_CP)
    #print("Number of received samples in the Freq domain: ", Received_OFDM_Freq.shape)

    # Channel estimation
    H_est = Channel_Est(Received_OFDM_Freq, ofdm)
    #print("\n Number of samples in Channel Estimate", H_est.shape)

    # i = 8
    # fig, ax = plt.subplots()
    # ax.plot(ofdm.total_Set, abs(channel.H), label='Correct_Channel')
    # ax.plot(ofdm.total_Set,abs(H_est[i,:]), label='Interpolation Est')
    # ax.legend(fontsize=10)
    # ax.set_xlabel('Carrier')
    # ax.set_ylabel('$|H(f)|$')
    # plt.savefig('Channel_estimate.png')

    # Channel Equalization
    Equalized_OFDM = Channel_Rem(Received_OFDM_Freq, H_est)
    print("\n Equalize the received OFDM symbols ...")
    #print("Number of received & Equalized samples is: ", Equalized_OFDM.shape)

    # Extract Payloads
    Est_Symbols = Extract_payloads(Equalized_OFDM, ofdm)
    #print("\n Shape of received & extracted payloads is: ", Est_Symbols.shape)

    # i = 0
    # fig, ax = plt.subplots()
    # x = Est_Symbols[:,:]
    # ax.plot(x.real, x.imag, 'bo')
    # ax.set_xlabel('Real Comp')
    # ax.set_ylabel('Imaginary Comp')
    # ax.set_title('Recv Constellation')
    # plt.savefig('Received_Contellation.png')

    # Payload Demodulation
    Demod_symbols = Demodulation(Est_Symbols, Mapper)
    print("\n size of demodulated payloads is: ", Demod_symbols.shape)

    # Parallel to serial
    Rcv_bit_stream = Parallel_to_Serial(Demod_symbols)
    #print("\n size of the received bit stream: ", Rcv_bit_stream.shape)

    return Rcv_bit_stream