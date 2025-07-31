# Transmit_PHY_OFDM.py
# --------------------
# Transmitter using a neural-network–based mapper

import numpy as np
from bit_loader import bit_stream_loader
from Modulation.Modulation import Serial_to_Parallel, OFDM_symbol, OFDM_time, WithCP
from Modulation.Channel import Channel, Apply_Channel, Channel_Est, Channel_Rem
from tensorflow.keras.models import load_model  # for loading the trained 4-bit→(I,Q) mapper
from pilotnn import build_pilot ##Pilot neural network

# Number of neurons in the mapper’s hidden layers (unused here, since we load a trained model)
num_neurons = 64
m_bits=16
k_bits=16
pilot_bins=8##Pilot bins
def get_full_mapper(Mapper):
    """
    Loads the small 4-bit→(I,Q) neural network that was trained and saved as 'mapper_nn_<Scheme>.h5'.
    The class name (e.g. "QAM_16") must match the SCHEME used during training.
    """
    filename = f"mapper_nn_{Mapper.__class__.__name__}.h5"
    return load_model(filename, compile=False)

def Send_OFDM(file_name, Mapper, ofdm):
    """
    Transmit OFDM symbols using a neural-network–based mapper.

    Args:
        file_name (str): Path to the file containing the raw bit stream (one continuous line of bits).
        Mapper:         Modulation object (e.g. QAM_16) with attributes .D (num data subcarriers) and .mu (bits per symbol).
        ofdm:           OFDM configuration object (provides K, CP, pilotValue, data_carriers, etc.).

    Returns:
        Tx_bit_stream (np.ndarray): Original 1D bit array (0/1).
        OFDM_CP (np.ndarray): Time-domain OFDM waveform with cyclic prefix, shape (num_OFDM_symbols, K+CP).
    """
    # 1) Load raw bitstream from file
    Tx_bit_stream, _ = bit_stream_loader(file_name)
    print(f"Transmitting a total of {len(Tx_bit_stream)} bits")

    # 2) Serial → Parallel conversion
    N_sym, bits_SP = Serial_to_Parallel(Tx_bit_stream, Mapper)
    # bits_SP.shape == (N_sym, Mapper.D, Mapper.mu)
    print(f"Number of symbols: {N_sym}, bits_SP shape: {bits_SP.shape}")

    # 3) NN-based mapping (4-bit→(I,Q) mapper, applied to each subcarrier)
    sub_mapper = get_full_mapper(Mapper)  # loads the small model expecting input shape (None, Mapper.mu)

    # Flatten each subcarrier’s 4 bits so we can batch‐predict
    N_sym, D, mu = bits_SP.shape  # D = number of data subcarriers, mu = bits per symbol (e.g. 4 for QAM_16)
    flat_bits = bits_SP.reshape(N_sym * D, mu)  # shape → (N_sym * D, mu)

    # Predict I/Q for each 4-bit chunk in one batch
    flat_iq = sub_mapper.predict(flat_bits)     # shape → (N_sym * D, 2)

    # Reshape back to (N_sym, D, 2)
    mapped_IQ = flat_iq.reshape(N_sym, D, 2)

    # Split I and Q components
    I_part = mapped_IQ[..., 0]  # shape → (N_sym, D)
    Q_part = mapped_IQ[..., 1]  # shape → (N_sym, D)

    # Reconstruct complex symbols: I + j*Q
    Mapped_Symbols = I_part + 1j * Q_part

    # 4) Assemble OFDM symbols
    OFDM_Symbols = OFDM_symbol(Mapped_Symbols, ofdm)  # shape → (N_sym, K)
    OFDM_Time    = OFDM_time(OFDM_Symbols)             # shape → (N_sym, K)
    OFDM_CP      = WithCP(OFDM_Time, ofdm)             # shape → (N_sym, K + CP)

    return Tx_bit_stream, OFDM_CP
