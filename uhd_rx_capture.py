#!/usr/bin/env python3
import argparse, time
import numpy as np
import uhd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", default="type=b200", help="UHD device args")
    ap.add_argument("--rate", type=float, default=5e6)
    ap.add_argument("--freq", type=float, default=915e6)
    ap.add_argument("--gain", type=float, default=10.0)
    ap.add_argument("--ant", default="RX2")
    ap.add_argument("--nsamps", type=int, default=300000, help="Target samples to collect")
    ap.add_argument("--timeout", type=float, default=3.0, help="Max wall time (s) to keep streaming")
    ap.add_argument("--outfile", default="rx_iq.fc32")
    args = ap.parse_args()

    print("[RX] Creating USRP...")
    usrp = uhd.usrp.MultiUSRP(args.args)

    # Antenna (best effort)
    try:
        usrp.set_rx_antenna(args.ant)
        print(f"[RX] Using antenna: {args.ant}")
    except Exception:
        pass

    print(f"[RX] Setting rate to {args.rate/1e6:.2f} Msps")
    usrp.set_rx_rate(args.rate)

    print(f"[RX] Tuning to {args.freq/1e6:.3f} MHz")
    try:
        usrp.set_rx_freq(uhd.types.TuneRequest(args.freq))
    except AttributeError:
        usrp.set_rx_freq(args.freq)

    print(f"[RX] Setting gain to {args.gain} dB")
    usrp.set_rx_gain(args.gain)

    # Stream setup
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    try:
        st_args.channels = [0]
    except Exception:
        pass
    rx_stream = usrp.get_rx_stream(st_args)

    # Figure out the right StreamCMD and StreamMode symbols across UHD versions
    try:
        StreamCMD = uhd.types.StreamCMD
    except AttributeError:
        StreamCMD = uhd.types.StreamCmd  # older UHD

    # Resolve StreamMode enums (both old/new names)
    Mode = getattr(uhd.types, "StreamMode", None)
    if Mode is not None:
        START_CONT = getattr(Mode, "start_cont", getattr(Mode, "START_CONTINUOUS", None))
        STOP_CONT  = getattr(Mode, "stop_cont",  getattr(Mode, "STOP_CONTINUOUS",  None))
        NUM_DONE   = getattr(Mode, "num_done",   getattr(Mode, "NUM_SAMPS_AND_DONE", None))
    else:
        # really old bindings fallback
        Mode = getattr(uhd.types, "stream_mode", None)
        START_CONT = getattr(Mode, "START_CONTINUOUS", None)
        STOP_CONT  = getattr(Mode, "STOP_CONTINUOUS",  None)
        NUM_DONE   = getattr(Mode, "NUM_SAMPS_AND_DONE", None)

    # If continuous symbols don’t exist, we can fall back to a single num_done grab
    use_continuous = START_CONT is not None and STOP_CONT is not None

    N = int(args.nsamps)
    buff = np.zeros(N, dtype=np.complex64)
    md = uhd.types.RXMetadata()

    if use_continuous:
        # Continuous streaming loop
        try:
            cmd_start = StreamCMD(START_CONT)
        except TypeError:
            # Some builds require specifying the enum as a kwarg
            cmd_start = StreamCMD(stream_mode=START_CONT)
        cmd_start.stream_now = True
        rx_stream.issue_stream_cmd(cmd_start)

        print(f"[RX] Starting continuous capture for up to {N} samples (timeout {args.timeout:.1f}s)...")
        got = 0
        t0 = time.time()
        per_recv = 8192

        while got < N and (time.time() - t0) < args.timeout:
            take = min(per_recv, N - got)
            view = buff[got:got+take]
            n = rx_stream.recv([view], md, 1.0)  # per-call timeout
            if n > 0:
                got += n

        try:
            cmd_stop = StreamCMD(STOP_CONT)
        except TypeError:
            cmd_stop = StreamCMD(stream_mode=STOP_CONT)
        rx_stream.issue_stream_cmd(cmd_stop)

        if got <= 0:
            print("[RX] No samples received.")
            return
        buff[:got].tofile(args.outfile)
        print(f"[RX] Captured {got} samples to {args.outfile}")

    else:
        # Fallback: one-shot NUM_DONE (works everywhere)
        print(f"[RX] Starting capture for {N} samples (num_done)...")
        try:
            cmd = StreamCMD(NUM_DONE)
        except TypeError:
            cmd = StreamCMD(stream_mode=NUM_DONE)
        cmd.num_samps = N
        cmd.stream_now = True
        rx_stream.issue_stream_cmd(cmd)

        # Give generous timeout so we don’t cut off early
        num_rx = rx_stream.recv([buff], md, 5.0 + N/args.rate)
        if num_rx <= 0:
            print("[RX] No samples received.")
            return
        buff[:num_rx].tofile(args.outfile)
        print(f"[RX] Captured {num_rx} samples to {args.outfile}")

if __name__ == "__main__":
    main()
