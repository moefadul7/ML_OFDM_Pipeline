#!/usr/bin/env python3
import numpy as np
import uhd

# Match TX settings
FS = 5e6          # sample rate (Hz)
FC = 915e6        # center frequency (Hz)
GAIN = 20.0       # RX gain (adjust as needed)
NSAMPS = 40000    # number of samples to capture
OUTFILE = "rx_iq.fc32"


def main():
    print("[RX] Creating USRP...")
    usrp = uhd.usrp.MultiUSRP("type=b200")

    print(f"[RX] Setting rate to {FS/1e6:.2f} Msps")
    usrp.set_rx_rate(FS)

    print(f"[RX] Tuning to {FC/1e6:.3f} MHz")
    # Handle both older/newer UHD Python bindings
    try:
        tune_req = uhd.types.TuneRequest(FC)
        usrp.set_rx_freq(tune_req)
    except AttributeError:
        usrp.set_rx_freq(FC)

    print(f"[RX] Setting gain to {GAIN} dB")
    usrp.set_rx_gain(GAIN)

    # One RX channel: CPU format = fc32, wire format = sc16
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(st_args)

    N = int(NSAMPS)
    buff = np.zeros(N, dtype=np.complex64)
    buffers = [buff]
    md = uhd.types.RXMetadata()

    print(f"[RX] Starting capture for {N} samples...")
    cmd = uhd.types.StreamCmd(uhd.types.StreamMode.num_done)
    cmd.num_samps = N
    cmd.stream_now = True
    rx_stream.issue_stream_cmd(cmd)

    num_rx = rx_stream.recv(buffers, md, 5.0)

    if num_rx <= 0:
        print("[RX] No samples received.")
        return

    buff[:num_rx].tofile(OUTFILE)
    print(f"[RX] Captured {num_rx} samples to {OUTFILE}")


if __name__ == "__main__":
    main()
