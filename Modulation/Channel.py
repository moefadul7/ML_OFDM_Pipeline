# This module includes classes and functions to Simulate a multipath wireless channel
import numpy as np
from scipy.interpolate import interp1d

class Channel:
    def __init__(self, K, SNR):
        self.h = np.array([1, 0, 0.1+0.1j])
        self.H = np.fft.fft(self.h, K)
        self.K = K
        self.SNR = SNR   # in dB


def Apply_Channel(signals, channel, With_Multipath):
    filtered_signals = []
    for signal in signals:
        if With_Multipath:
            filtered = np.convolve(signal,channel.h)
        else:
            filtered = signal  ##

        signal_power = np.mean(abs(filtered ** 2))
        noise_sigma = signal_power * 10 ** (-channel.SNR / 10)
        #print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, noise_sigma))

        # Generate and scale noise
        noise = np.sqrt(noise_sigma / 2) * (np.random.randn(*filtered.shape) + 1j * np.random.randn(*filtered.shape))
        filtered_signals.append(filtered + noise)
    return np.array(filtered_signals)

def Channel_Est(Received_OFDM_Freq, ofdm):
    Channel_Estimates = []
    for signal in Received_OFDM_Freq:
        pilots = signal[ofdm.pilots]
        Pilot_H = pilots/ofdm.pilotValue

        # Interpolate
        H_abs = interp1d(ofdm.pilots, abs(Pilot_H), kind='cubic')(ofdm.total_Set)
        H_ph = interp1d(ofdm.pilots, np.angle(Pilot_H), kind='cubic')(ofdm.total_Set)
        H = H_abs * np.exp(1j * H_ph)
        Channel_Estimates.append(H)
    return np.array(Channel_Estimates)

def Channel_Rem(Received_OFDM_Freq, H_est):
    return Received_OFDM_Freq / H_est