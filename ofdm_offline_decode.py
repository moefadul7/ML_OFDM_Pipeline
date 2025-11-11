#!/usr/bin/env python3
import numpy as np
from Modulation.Modulation import OFDM  # uses your existing OFDM class

# Match TX settings
K = 64
CP = 16
P = 8
PILOTVAL = 1 + 0j
MU = 2  # QPSK

def qpsk_hard_demap(symbols):
    bits = np.zeros(symbols.size * 2, dtype=np.int8)
    bits[0::2] = (np.real(symbols) > 0).astype(np.int8)
    bits[1::2] = (np.imag(symbols) > 0).astype(np.int8)
    return bits

def main():
    # 1) Load TX bits and RX samples
    tx_bits = np.load("tx_bits.npy")
    rx = np.fromfile("rx_iq.fc32", dtype=np.complex64)
    if rx.size == 0:
        print("No RX samples found in rx_iq.fc32")
        return

    # 2) Recreate OFDM layout
    ofdm = OFDM(K=K, P=P, CP=CP, pilotValue=PILOTVAL)
    D = len(ofdm.data_carriers)
    bits_per_sym = D * MU

    # 3) Infer number of OFDM symbols from TX bits
    Nsym_tx = tx_bits.size // bits_per_sym
    if Nsym_tx == 0:
        print("Not enough TX bits for one OFDM symbol.")
        return

    frame_len = Nsym_tx * (K + CP)
    if rx.size < frame_len:
        # If capture is short, truncate to what fits
        Nsym = rx.size // (K + CP)
        frame_len = Nsym * (K + CP)
    else:
        Nsym = Nsym_tx

    if Nsym == 0:
        print("Not enough RX samples for one OFDM symbol.")
        return

    # 4) Crude timing sync: slide energy window of frame_len, pick max
    power = np.convolve(np.abs(rx)**2, np.ones(frame_len), mode="valid")
    start = np.argmax(power)
    rx_frame = rx[start:start+frame_len]

    # 5) Reshape into OFDM symbols and remove CP
    rx_blocks = rx_frame.reshape(Nsym, K + CP)
    rx_no_cp = rx_blocks[:, CP:]              # (Nsym, K)

    # 6) FFT to go to frequency domain
    RX_F = np.fft.fft(rx_no_cp, n=K, axis=1)  # (Nsym, K)

    # 7) Channel estimate from pilots (avg over symbols)
    pilots_idx = ofdm.pilots
    pilots_rx = RX_F[:, pilots_idx]           # (Nsym, Npilots)
    H_pilots = pilots_rx / PILOTVAL
    H_pilots_mean = np.mean(H_pilots, axis=0)

    # 8) Interpolate H across all subcarriers
    all_idx = np.arange(K)
    H_real = np.interp(all_idx, pilots_idx, np.real(H_pilots_mean))
    H_imag = np.interp(all_idx, pilots_idx, np.imag(H_pilots_mean))
    H_est = H_real + 1j * H_imag

    # 9) Equalize
    RX_eq = RX_F / (H_est[None, :] + 1e-12)

    # 10) Pull data subcarriers
    data_eq = RX_eq[:, ofdm.data_carriers]    # (Nsym, D)
    data_flat = data_eq.reshape(-1)

    # 11) Hard QPSK demap
    rx_bits_hat = qpsk_hard_demap(data_flat)

    # 12) Compute BER / BLER vs TX bits
    L = min(tx_bits.size, rx_bits_hat.size)
    tx_use = tx_bits[:L]
    rx_use = rx_bits_hat[:L]

    bit_errors = np.sum(tx_use != rx_use)
    ber = bit_errors / L if L > 0 else float("nan")

    blocks = L // bits_per_sym
    if blocks > 0:
        tx_blk = tx_use[:blocks*bits_per_sym].reshape(blocks, bits_per_sym)
        rx_blk = rx_use[:blocks*bits_per_sym].reshape(blocks, bits_per_sym)
        blk_err = np.any(tx_blk != rx_blk, axis=1)
        bler = np.mean(blk_err)
    else:
        bler = float("nan")

    print(f"Used {L} bits")
    print(f"BER  = {ber:.3e}")
    print(f"BLER = {bler:.3e}")

if __name__ == "__main__":
    main()

