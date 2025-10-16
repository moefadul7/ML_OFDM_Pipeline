import numpy as np
import matplotlib.pyplot as plt

# ====== Load IQ samples ======
# Replace with  actual IQ output file path
iq_data = np.fromfile('Alice_output_stream', dtype=np.complex64)

# ====== Separate I and Q ======
I = np.real(iq_data)
Q = np.imag(iq_data)

# ====== Plot Time Domain ======
plt.figure(figsize=(10, 4))
plt.plot(I, label='In-phase (I)')
plt.plot(Q, label='Quadrature (Q)')
plt.title('IQ Samples - Time Domain')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_domain.png')
plt.close()

# ====== Plot Frequency Domain (PSD) ======
sample_rate = 64000  # Change to match your SDR settings
plt.figure(figsize=(10, 4))
plt.psd(iq_data, NFFT=1024, Fs=sample_rate)
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig('psd.png')
plt.close()

# ====== FFT Spectrum Plot ======
fft_data = np.fft.fftshift(np.fft.fft(iq_data))
freq_axis = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/sample_rate))

plt.figure(figsize=(10, 4))
plt.plot(freq_axis, 20*np.log10(np.abs(fft_data)))
plt.title('FFT Spectrum of IQ Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.tight_layout()
plt.savefig('fft_spectrum.png')
plt.close()

# ====== Constellation Plot ======
plt.figure(figsize=(6, 6))
plt.scatter(I, Q, color='blue', s=2, alpha=0.5)
plt.title('Constellation Diagram')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('constellation.png')
plt.close()

print(" Plots saved as time_domain.png, psd.png, fft_spectrum.png, and constellation.png")
