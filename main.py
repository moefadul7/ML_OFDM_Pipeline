#FZB297@mocs.utc.edu 
import sys
import os
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
import matplotlib.ticker as ticker 

# ========================================================================
# Parameter Setup
# ========================================================================

m_bits = 16
k_bits = 16 ##Are these 15?
c_bits = 16
pad = 'same'

m_train = 2 ** m_bits 
n_epoch = 150  
batch_size = 256
n_batches = m_train // batch_size
abecycles = 1 
evecycles = 1  
n_samples = 100
Train = False    

model_file = 'Alice_Bob_Model.keras'

K = 64
CP = 16
P = 8
pilotValue = 3 + 3j
With_Multipath = True

file_name_A = 'Alice_output_stream'
file_name_B = 'Bob_input_stream'

if len(sys.argv) != 4:
    print("You need to provide 3 configurable parameters:\n"
          "1. Modulation per Carrier\n"
          "2. Q bits\n"
          "3. SNR")
    exit()

Mod_Per_Carrier = sys.argv[1]
num_bits = int(sys.argv[2])
SNR_range = range(2, 26, 3)

# Containers for results
snr_values = []
bler_values = []
ber_values = []

# ========================================================================
# Helper Functions
# ========================================================================

def Mapping(scheme, D):
    if scheme == "QPSK":
        return QPSK(D)
    elif scheme == "QAM_16":
        return QAM_16(D)
    elif scheme == "QAM_64":
        return QAM_64(D)
    else:
        raise ValueError(f"Unsupported modulation scheme: {scheme}")

def soft_thresholding(predictions, threshold=0.5, margin=0.1):
    return np.where(predictions > (threshold + margin), 1,
                    np.where(predictions < (threshold - margin), 0, predictions))

# ========================================================================
# Train Once
# ========================================================================

if Train:
    Alice_Model, Bob_Model, Eve_Model, abmodel, evemodel = NN_setup(m_bits, k_bits, c_bits)
    framework_train(m_bits, k_bits, n_batches, batch_size, n_epoch,
                    Alice_Model, Bob_Model, abmodel, evemodel,
                    abecycles, evecycles, n_samples, model_file)

# ========================================================================
# Main Evaluation Loop
# ========================================================================

for SNR in SNR_range:
    print(f"\n--- Running experiment for SNR = {SNR} dB ---")

    # Alice quantization
    Alice_out_quantized, Alice_out_stream, Alice_out, tst_msg, tst_key = Alice_Load_Quantize(
        model_file, n_samples, num_bits
    )

    print(f"Shape of tst_msg: {tst_msg.shape}")
    print(f"Shape of tst_key: {tst_key.shape}")

    # OFDM setup and transmission
    ofdm = OFDM(K, P, CP, pilotValue)
    Mapper = Mapping(Mod_Per_Carrier, len(ofdm.data_carriers))
    Tx_bit_stream, OFDM_CP = Send_OFDM(file_name_A, Mapper, ofdm)

    print("Applying Wireless Channel Response ...")
    channel = Channel(ofdm.K, SNR)
    Received_Signals = Apply_Channel(OFDM_CP, channel, With_Multipath)

    print("Collecting OFDM symbols by the receiver ...")
    Rcv_bit_stream = Recv_OFDM(Received_Signals, ofdm, Mapper)

    # BER calculation
    ber_value, span = BER(Tx_bit_stream, Rcv_bit_stream)
    print(f"BER for transmitted stream of length {len(Tx_bit_stream)}: {ber_value}")

    # Dump received stream for Bob
    Dump_Rcvd_stream(file_name_B, Rcv_bit_stream, span)

    # Bob dequantization
    Bob_in, Bob_out = Bob_Load_Dequantize(model_file, file_name_B, tst_key)
    Bob_out_bin = np.where(Bob_out < 0.5, 0, 1)

    # BLER calculation
    Error_Per_Msg = np.where(np.sum(np.abs(tst_msg - Bob_out_bin), axis=1) > 0, 1, 0)
    BLER = np.sum(Error_Per_Msg)
    print(f"Total End-to-End BLER over {n_samples} messages: {BLER}")

    # Save results
    snr_values.append(SNR)
    ber_values.append(ber_value)
    bler_values.append(BLER)

# ========================================================================
# Plotting Results
# ========================================================================

# BER vs SNR
plt.figure(figsize=(10, 6))
plt.plot(snr_values, ber_values, marker='o', label='BER')
plt.title('BER vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate')
plt.grid(True)
plt.xticks(snr_values)
plt.legend()
plt.tight_layout()
plt.savefig('ber_vs_snr.png')
plt.show()

# BLER vs SNR
bler_rates = [b / n_samples for b in bler_values]
plt.figure(figsize=(10, 6))
plt.plot(snr_values, bler_rates, marker='o', color='red', label='BLER')
plt.title('BLER vs SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('Block Error Rate')
plt.grid(True)
plt.xticks(snr_values)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('bler_vs_snr.png')
plt.show()

# Log-scale BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, ber_values, marker='o', label='BER')
plt.title('BER vs SNR (log-scale)')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (log scale)')
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.xticks(snr_values)
plt.legend()
plt.tight_layout()
plt.savefig('ber_vs_snr_log.png')
plt.show()

# Log-scale BLER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, bler_rates, marker='o', color='red', label='BLER')
plt.title('BLER vs SNR (log-scale)')
plt.xlabel('SNR (dB)')
plt.ylabel('Block Error Rate (log scale)')
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.xticks(snr_values)
plt.legend()
plt.tight_layout()
plt.savefig('bler_vs_snr_log.png')
plt.show()  

import matplotlib.ticker as ticker

plt.figure(figsize=(10, 6))

# Prevent log(0)
bler_rates_safe = [b if b > 0 else 1e-4 for b in bler_rates]

plt.plot(snr_values, bler_rates_safe, marker='o', color='red', label='Offline-QPSK')
plt.yscale('log')

# Custom ticks (IEEE-style base-10 format)
ax = plt.gca()
ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4])
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Block Error Rate (BLER)', fontsize=12)
plt.title('BLER vs SNR (log scale)', fontsize=14)
plt.xticks(snr_values) 
plt.legend()
plt.tight_layout()
plt.savefig("bler_vs_snr_log_styled.png", dpi=300)
plt.show()



  
# Log-scale BLER vs SNR (styled to match IEEE paper format)

