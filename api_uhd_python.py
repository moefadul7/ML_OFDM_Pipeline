"""
UHD-based OFDM QPSK transmitter (802.11a-like framing) that plugs into your
existing codebase (bit_loader / Modulation / optional Alice).
"""

import os, sys, time, importlib
import numpy as np
from importlib.machinery import SourceFileLoader

# ---------- lock to this file's folder so relative files resolve ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------- flexible module loaders (handle package folder or .py file) ----------
def _load_module_flex(modname: str, candidates):
    """
    Try multiple candidate paths relative to SCRIPT_DIR. The first existing file wins.
    """
    for cand in candidates:
        path = os.path.join(SCRIPT_DIR, cand)
        if os.path.exists(path):
            return SourceFileLoader(modname, path).load_module()
    raise FileNotFoundError(f"Couldn't find module {modname}. Looked for: {candidates}")

def _load_modulation():
    """
    Load Modulation whether it's a package dir (with an __init__.py or a main .py)
    or a single file in the project root.
    """
    pkg_dir = os.path.join(SCRIPT_DIR, "Modulation")
    if os.path.isdir(pkg_dir):
        # Try to import as a package first
        try:
            return importlib.import_module("Modulation")
        except Exception:
            # Fall back to direct file loads inside the folder
            inner_candidates = [
                "Modulation/__init__.py",
                "Modulation/Modulation.py",
                "Modulation/modulation.py",
                "Modulation/OFDM.py",
                "Modulation/ofdm.py",
            ]
            return _load_module_flex("Modulation", inner_candidates)
    # Not a folder: try root-level files
    return _load_module_flex("Modulation", ["Modulation.py", "modulation.py"])

MOD  = _load_modulation()
BIT5 = _load_module_flex("bit_loader5", ["bit_loader.py", "bit_loader(5).py"])
ALICE = None  # will be loaded lazily only if we use the Alice model

# UHD (pyuhd)
import uhd

# ---------------- RF / PHY ----------------
CENTER_FREQUENCY = 2.412e9   # Hz (use cabled/attenuated or legal band)
TX_RATE          = 20e6      # S/s
TX_GAIN          = 10        # dB (start low)
K                = 64        # FFT size
CP_LEN           = 16        # CP length
P                = 8         # pilot plan used by your Modulation.OFDM
PILOT_VALUE      = 3+3j      # fixed pilot value
FRAME_REPEAT     = 10        # repeat frame N times

# ------------- bit sources ----------------
USE_ALICE_MODEL  = False
ALICE_MODEL      = "alice_quantization_model.h5"
ALICE_NSAMP      = 100
BIT_FILE         = "Alice_output_stream"   # one line of 0/1 chars

# -------- helpers: bit sources ------------
def get_bits_from_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bit file not found: {path}")
    with open(path, "r") as f:
        s = f.read().strip()
    return np.fromiter((1 if ch == '1' else 0 for ch in s), dtype=np.int8)

def get_bits_via_alice(model_path: str, n_samples: int) -> np.ndarray:
    global ALICE
    if ALICE is None:
        ALICE = _load_module_flex("Alice", ["Alice.py", "Alice(3).py"])
    # Alice_Load_Quantize(model_path, n_samples, 8) -> (..., Alice_out_stream, ...)
    _, alice_stream, _, _, _ = ALICE.Alice_Load_Quantize(model_path, n_samples, 8)
    return np.fromiter((1 if ch == '1' else 0 for ch in alice_stream), dtype=np.int8)

# ----- build OFDM payload waveform --------
def build_ofdm_waveform(bits: np.ndarray,
                        k: int = K,
                        p: int = P,
                        cp: int = CP_LEN,
                        pilot_val: complex = PILOT_VALUE) -> np.ndarray:
    ofdm = MOD.OFDM(K=k, P=p, CP=cp, pilotValue=pilot_val)
    D = len(ofdm.data_carriers)
    qpsk = MOD.QPSK(D)
    n_syms, bits_sp = MOD.Serial_to_Parallel(bits, qpsk)
    mapped = MOD.Bits_to_Symbols(bits_sp, qpsk)     # [n_syms, D]
    fd_syms = MOD.OFDM_symbol(mapped, ofdm)         # [n_syms, K]
    td_syms = MOD.OFDM_time(fd_syms)                # [n_syms, K]
    td_cp   = MOD.WithCP(td_syms, ofdm)             # [n_syms, K+CP]
    tx      = td_cp.reshape(-1).astype(np.complex64)
    peak = np.max(np.abs(tx)) or 1.0
    tx /= peak
    return tx

# ------------- simple preamble ------------
def build_preamble_like_lts(k: int = K) -> np.ndarray:
    freq = np.zeros(k, dtype=np.complex64)
    bins = np.hstack([np.arange(-26, 0), np.arange(1, 27)])
    for i, b in enumerate(bins):
        freq[b % k] = 1 if (i % 2 == 0) else -1
    td = np.fft.ifft(np.fft.ifftshift(freq)).astype(np.complex64)
    return np.hstack([td, td])

# --------------- UHD TX -------------------
def uhd_tx(iq: np.ndarray,
           center_freq: float = CENTER_FREQUENCY,
           rate: float = TX_RATE,
           gain: float = TX_GAIN,
           args: str = "") -> int:
    usrp = uhd.usrp.MultiUSRP(args)
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.types.tune_request(center_freq))
    usrp.set_tx_gain(gain)

    try:
        st_args = uhd.stream_args('fc32')
    except TypeError:
        st_args = uhd.usrp.StreamArgs(cpu_format="fc32", otw_format="sc16")
    tx_stream = usrp.get_tx_stream(st_args)

    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst   = False
    md.has_time_spec  = False

    CHUNK = 32768
    total_sent = 0
    for i in range(0, len(iq), CHUNK):
        total_sent += tx_stream.send(iq[i:i+CHUNK], md)
        md.start_of_burst = False

    md.end_of_burst = True
    tx_stream.send(np.zeros(0, dtype=np.complex64), md)
    return total_sent

# ----------------- main -------------------
if __name__ == "__main__":
    # choose a bit source
    if USE_ALICE_MODEL and os.path.exists(ALICE_MODEL):
        bits = get_bits_via_alice(ALICE_MODEL, ALICE_NSAMP)
    elif os.path.exists(BIT_FILE):
        bits = get_bits_from_file(BIT_FILE)
    else:
        tx_bits, _ = BIT5.bit_stream_loader('test_messages')
        bits = tx_bits.astype(np.int8)

    preamble = build_preamble_like_lts(K)
    payload  = build_ofdm_waveform(bits, k=K, p=P, cp=CP_LEN, pilot_val=PILOT_VALUE)

    one_frame = np.hstack([preamble, payload]).astype(np.complex64)
    iq        = np.tile(one_frame, FRAME_REPEAT).astype(np.complex64)

    print(f"IQ length: {len(iq)} | rate: {TX_RATE/1e6:.1f} Msps | pilot={PILOT_VALUE}")
    n_sent = uhd_tx(iq)
    print(f"Sent {n_sent} samples")
