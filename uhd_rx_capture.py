#!/usr/bin/env python3
import numpy as np
import uhd

# ==== Match TX settings ====
FS = 5e6            # sample rate (Hz)
FC = 915e6          # center frequency (Hz)
GAIN = 10.0         # RX gain (dB)
NSAMPS = 200000     # capture more to make sync easy
OUTFILE = "rx_iq.fc32"

# Let UHD auto-pick the device; set to "serial=XXXXXXX" if you want to pin it
DEVICE_ARGS = ""


def _make_stream_cmd(N: int):
    """Build a UHD StreamCMD compatible with multiple UHD versions."""
    # StreamCMD vs StreamCmd
    try:
        StreamCMD = uhd.types.StreamCMD
    except AttributeError:
        StreamCMD = uhd.types.StreamCmd

    # Mode name varies by version
    try:
        mode = uhd.types.StreamMode.num_samples_and_done
    except AttributeError:
        try:
            mode = uhd.types.StreamMode.num_done
        except AttributeError:
            # Older/alternate naming
            mode = uhd.types.StreamMode.NUM_SAMPS_AND_DONE

    cmd = StreamCMD(mode)
    cmd.num_samps = int(N)
    cmd.stream_now = True
    return cmd


def main():
    print("[RX] Creating USRP...")
    usrp = uhd.usrp.MultiUSRP(DEVICE_ARGS)

    # Use RX2 (receive-only) on B200mini if available
    try:
        usrp.set_rx_antenna("RX2")
        print("[RX] Using antenna: RX2")
    except Exception:
        print("[RX] Antenna selection not supported; continuing.")

    print(f"[RX] Setting rate to {FS/1e6:.2f} Msps")
    usrp.set_rx_rate(FS)

    print(f"[RX] Tuning to {FC/1e6:.3f} MHz")
    try:
        usrp.set_rx_freq(uhd.types.TuneRequest(FC))
    except AttributeError:
        usrp.set_rx_freq(FC)

    print(f"[RX] Setting gain to {GAIN} dB")
    usrp.set_rx_gain(GAIN)

    # One RX channel: CPU=fc32, wire=sc16
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(st_args)

    N = int(NSAMPS)
    buff = np.zeros(N, dtype=np.complex64)
    buffers = [buff]
    md = uhd.types.RXMetadata()

    print(f"[RX] Starting capture for {N} samples...")
    cmd = _make_stream_cmd(N)
    rx_stream.issue_stream_cmd(cmd)

    num_rx = rx_stream.recv(buffers, md, 5.0)  # 5s timeout

    if num_rx <= 0:
        print("[RX] No samples received.")
        return

    buff[:num_rx].tofile(OUTFILE)
    print(f"[RX] Captured {num_rx} samples to {OUTFILE}")


if __name__ == "__main__":
    main()
