# This script is used to configure and launch OFDM transmitter as well as apply channel response
import numpy as np
import matplotlib.pyplot as plt
from bit_loader import bit_stream_loader
from Modulation.Modulation import  QAM_16, Serial_to_Parallel, Parallel_to_Serial, Bits_to_Symbols, Demodulation
from Modulation.Modulation import OFDM, OFDM_symbol, OFDM_time, WithCP, WithoutCP, OFDM_freq, Extract_payloads, BER
from Modulation.Channel import Channel, Apply_Channel, Channel_Est, Channel_Rem


def Send_OFDM(file_name, Mapper, ofdm):
    # Load bit-stream
    Tx_bit_stream, bit_stream = bit_stream_loader(file_name)
    print("Transmitting a total of {} samples\n".format(len(Tx_bit_stream)))

    # Serial to Parallel Conversion
    N_symbols_to_send, bits_SP = Serial_to_Parallel(Tx_bit_stream, Mapper)
    print("Number of symbols to send {}, Dimensions {}".format(N_symbols_to_send, bits_SP.shape))
    # # Mapping to symbols
    Mapped_Symbols = Bits_to_Symbols(bits_SP, Mapper)

    # Assemble OFDM symbols
    OFDM_Symbols = OFDM_symbol(Mapped_Symbols, ofdm)
    #print("Number of OFDM carriers in frequency domain: ", OFDM_Symbols.shape)

    # Transform to time domain
    OFDM_Time = OFDM_time(OFDM_Symbols)
    #print("Number of OFDM samples in time-domain before CP: ", OFDM_Time.shape)

    # Add Cyclic Prefix
    OFDM_CP = WithCP(OFDM_Time,ofdm)
    #print("Number of samples per OFDM symbol in time domain with CP: ", OFDM_CP.shape[1])

    return Tx_bit_stream, OFDM_CP
    #return N_symbols_to_send, bits_SP