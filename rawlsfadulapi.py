#!/usr/bin/env python3
"""
Single-file UHD OFDM TX that uses your Alice NN quantizer (./DL_DSSS/Alice.py)
and optional NN mapper (MapperNN.py). Designed for your ML_OFDM_Pipeline repo.
"""

import os
import sys
import time
import argparse
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

# ---------------- lock CWD & sys.path ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _load_module(modname: str, relpath: str):
    path = os.path.join(SCRIPT_DIR, relpath)
    if not os.path.exists(path):
        raise ModuleNotFoundError(f"Missing {relpath}")
    spec = spec_from_file_location(modname, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# ---------------- Alice (NN quantizer) ----------------
try:
    Alice_mod = _load_module("Alice", os.path.join("DL_DSSS", "Alice.py"))
    Alice_Load_Quantize = getattr(Alice_mod, "Alice_Load_Quantize")
except Exception as e:
    raise SystemExit(f"Could not load Alice from ./DL_DSSS/Alice.py. Error: {e}")


# ---------------- Modulation (OFDM + LUT mappers) ----------------
try:
    from Modulation.Modulation import (
        OFDM,
        QPSK,
        QAM_16,
        QAM_64,
        Serial_to_Parallel,
        Bits_to_Symbols,
        OFDM_symbol,
        OFDM_time,
        WithCP,
    )
except ModuleNotFoundError:
    Modulation = _load_module("Modulation", "Modulation.py")
    OFDM = Modulation.OFDM
    QPSK = Modulation.QPSK
    QAM_16 = Modulation.QAM_16
    QAM_64 = Modulation.QAM_64
    Serial_to_Parallel = Modulation.Serial_to_Parallel
    Bits_to_Symbols = Modulation.Bits_to_Symbols
    OFDM_symbol = Modulation.OFDM_symbol
    OFDM_time = getattr(Modulation, "OFDM_time", getattr(Modulation, "OFDM_Time", None))
    if OFDM_time is None:
        raise AttributeError("Modulation.py is missing OFDM_time/OFDM_Time")
    WithCP = Modulation.WithCP


# ---------------- Optional NN mapper ----------------
_HAS_MAPPER_NN = True
try:
    from MapperNN import build_mapper_nn, build_full_mapper
except ModuleNotFoundError:
    try:
        MapperNN = _load_module("MapperNN", "MapperNN.py")
        build_mapper_nn = getattr(MapperNN, "build_mapper_nn", None)
        build_full_mapper = getattr(MapperNN, "build_full_mapper", None)
        if build_mapper_nn is None:
            _HAS_MAPPER_NN = False
    except Exception:
        _HAS_MAPPER_NN = False


# ---------------- UHD ----------------
try:
    import uhd
except ModuleNotFoundError as e:
    raise SystemExit("UHD Python bindings not found. Install UHD (python3-uhd). Error: %s" % e)


# ---------------- defaults ----------------
DEF_FS = 20e6
DEF_FC = 915e6
DEF_GAIN = 5
K, CP, P = 64, 16, 8
PILOTVAL = 1 + 0j


# ---------------- helpers ----------------
def get_bits_from_alice(model_path: str, n_samples: int, num_bits: int = 16) -> np.ndarray:
    A_bin, _, _, _, _ = Alice_Load_Quantize(model_path, n_samples, num_bits)
    return A_bin.astype(np.int8).reshape(-1)


def classic_map_bits(bits: np.ndarray, modulation: str, D: int):
    m = modulation.upper()
    if m == "QPSK":
        Mapper, mu = QPSK(D), 2
    elif m in ("QAM16", "QAM_16", "16QAM"):
        Mapper, mu = QAM_16(D), 4
    elif m in ("QAM64", "QAM_64", "64QAM"):
        Mapper, mu = QAM_64(D), 6
    else:
        raise ValueError("Unsupported modulation: " + modulation)

    Nsym, bits_SP = Serial_to_Parallel(bits.copy(), Mapper)
    syms = Bits_to_Symbols(bits_SP, Mapper).astype(np.complex64)
    return syms, mu


def nn_map_bits(bits: np.ndarray, modulation: str, D: int, weights_path: str):
    if not _HAS_MAPPER_NN:
        raise RuntimeError("MapperNN not available. Use --use_nn_mapper 0 or add MapperNN.py + weights.")
    if not (weights_path and os.path.exists(weights_path)):
        raise FileNotFoundError(f"Mapper weights not found: {weights_path}")

    m = modulation.upper()
    if m == "QPSK":
        mu = 2
    elif m in ("QAM16", "QAM_16", "16QAM"):
        mu = 4
    elif m in ("QAM64", "QAM_64", "64QAM"):
        mu = 6
    else:
        raise ValueError("Unsupported modulation: " + modulation)

    model = build_mapper_nn(mu, 32)
    model.load_weights(weights_path)

    sym_bits = D * mu
    if bits.size < sym_bits:
        raise RuntimeError(f"Not enough bits ({bits.size}) for one OFDM symbol ({sym_bits}).")

    bits = bits[: bits.size - (bits.size % sym_bits)]
    Nsym = bits.size // sym_bits

    syms = np.zeros((Nsym, D), dtype=np.complex64)
    for i in range(Nsym):
        chunk = bits[i * sym_bits:(i + 1) * sym_bits].reshape(D, mu).astype(np.float32)
        out_iq = []
        for sc in range(D):
            y = model.predict(chunk[sc][None, ...], verbose=0)
            out_iq.append(y[0, 0] + 1j * y[0, 1])
        iq = np.array(out_iq, dtype=np.complex64)
        p = np.mean(np.abs(iq) ** 2) + 1e-12
        syms[i, :] = iq / np.sqrt(p)
    return syms, mu


def build_ofdm_burst(payload_syms, ofdm):
    F = OFDM_symbol(payload_syms, ofdm)
    td = OFDM_time(F)
    td_cp = WithCP(td, ofdm)
    burst = td_cp.reshape(-1).astype(np.complex64)
    burst /= (np.max(np.abs(burst)) + 1e-12)
    return burst


def uhd_tx(iq, fs, fc, gain, uhd_args: str = "", repeat: int = 1, save_iq: str = ""):
    usrp = uhd.usrp.MultiUSRP(uhd_args)
    usrp.set_tx_rate(fs)

    # Set TX frequency
    try:
        usrp.set_tx_freq(uhd.types.TuneRequest(fc))
    except AttributeError:
        usrp.set_tx_freq(fc)

    usrp.set_tx_gain(gain)

    # 1 TX channel: CPU=fc32, wire=sc16
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st = usrp.get_tx_stream(st_args)

    if save_iq:
        iq.astype(np.complex64).tofile(save_iq)
        print(f"Saved burst IQ to {save_iq} ({iq.size} complex samples)")

    # UHD Python expects list-of-arrays: one per channel
    buf = [iq.astype(np.complex64)]
    zbuf = [np.zeros(0, np.complex64)]

    for _ in range(repeat):
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = False
        st.send(buf, md)

        md.start_of_burst = False
        md.end_of_burst = True
        st.send(zbuf, md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", default="", help="UHD device args (e.g., 'type=b200')")
    ap.add_argument("--rate", type=float, default=DEF_FS)
    ap.add_argument("--freq", type=float, default=DEF_FC)
    ap.add_argument("--gain", type=float, default=DEF_GAIN)
    ap.add_argument("--mod", default="QPSK", choices=["QPSK", "QAM16", "QAM64"])
    ap.add_argument("--alice_model", required=True, help="Path to alice_quantization_model.h5")
    ap.add_argument("--n_alice_samples", type=int, default=200,
                    help="How many 16-bit messages Alice produces")
    ap.add_argument("--use_nn_mapper", type=int, default=0,
                    help="1=use MapperNN with --mapper_weights")
    ap.add_argument("--mapper_weights", default="",
                    help="Path to mapper_nn_<SCHEME>.h5")
    ap.add_argument("--save_iq", default="",
                    help="Optional path to save burst IQ (fc32)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat the same burst N times")
    args = ap.parse_args()

    fs, fc, gain = args.rate, args.freq, args.gain

    bits = get_bits_from_alice(args.alice_model, args.n_alice_samples, 16)

    ofdm = OFDM(K=K, P=P, CP=CP, pilotValue=PILOTVAL)
    D = len(ofdm.data_carriers)

    if args.use_nn_mapper:
        payload_syms, mu = nn_map_bits(bits, args.mod, D, args.mapper_weights)
    else:
        payload_syms, mu = classic_map_bits(bits, args.mod, D)

    sym_bits = D * mu
    if bits.size < sym_bits:
        raise SystemExit(
            f"Not enough bits ({bits.size}) for one OFDM symbol ({sym_bits}). "
            f"Increase --n_alice_samples."
        )

    burst = build_ofdm_burst(payload_syms, ofdm)

    print(f"Built frame: {burst.size} complex samples @ {fs/1e6:.2f} Msps")
    print(
        f"OFDM: K={K}, CP={CP}, data_subcarriers={D}, mod={args.mod}, "
        f"NN_mapper={bool(args.use_nn_mapper)}; UHD args='{args.args}'"
    )

    t0 = time.perf_counter()
    uhd_tx(burst, fs, fc, gain, uhd_args=args.args, repeat=args.repeat, save_iq=args.save_iq)
    print(f"TX done in {(time.perf_counter() - t0) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
bits = get_bits_from_alice(args.alice_model, args.n_alice_samples, 16)
np.save("tx_bits.npy", bits)
print(f"Saved TX bits to tx_bits.npy (len={bits.size})")
