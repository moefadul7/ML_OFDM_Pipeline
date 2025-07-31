#!/usr/bin/env python3
# train_mapper.py
# ----------------
# Usage: python3 train_mapper.py <SCHEME>
#   SCHEME ∈ { QPSK, QAM_16, QAM_64 }
#
# This script will:
#   1) Import the correct Modulation class (QPSK, QAM_16, or QAM_64).
#   2) Figure out mu = bits_per_symbol from that class.
#   3) Build all bit‐combinations of length mu.
#   4) Look up the corresponding (I,Q) from the static mapper dictionary.
#   5) Train an MLP to reproduce those (I,Q) pairs from each bit‐tuple.
#   6) Save the trained model as "mapper_nn_<SCHEME>.h5" for later loading.

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Import the three possible Modulation classes:
from Modulation.Modulation import QPSK, QAM_16, QAM_64
from MapperNN import build_mapper_nn

def main():
    parser = argparse.ArgumentParser(
        description="Train a neural‐network mapper for a given modulation scheme.")
    parser.add_argument(
        "scheme",
        choices=["QPSK", "QAM_16", "QAM_64"],
        help="Which modulation to train (e.g. QPSK, QAM_16, or QAM_64)."
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=32,
        help="Number of hidden neurons in each Dense layer of the mapper (default: 32)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs (default: 200)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: mu, i.e. bits_per_symbol)."
    )
    args = parser.parse_args()

    SCHEME = args.scheme
    NUM_NEURONS = args.neurons

    # 1) Select the correct Modulation class based on the argument:
    if SCHEME == "QPSK":
        ModClass = QPSK
    elif SCHEME == "QAM_16":
        ModClass = QAM_16
    else:  # SCHEME == "QAM_64"
        ModClass = QAM_64

    # 2) Instantiate with D=1 just to get mu (bits_per_symbol):
    mapper_ref = ModClass(D=1)
    mu = mapper_ref.mu        # e.g. 2 for QPSK, 4 for QAM_16, 6 for QAM_64
    bits_per_symbol = mu      # alias for clarity

    # 3) Build all possible bit‐tuples of length mu:
    all_bit_combinations = list(itertools.product([0, 1], repeat=mu))
    X = np.array(all_bit_combinations, dtype=np.float32)  # shape (2^mu, mu)

    # 4) Convert each bit‐tuple to its correct (I, Q) via the static mapper:
    Y = np.zeros((len(all_bit_combinations), 2), dtype=np.float32)
    for idx, bit_tuple in enumerate(all_bit_combinations):
        complex_point = mapper_ref.mapper[tuple(bit_tuple)]
        Y[idx, 0] = complex_point.real
        Y[idx, 1] = complex_point.imag

    # 5) Build and compile the MLP (mu → 2):
    model = build_mapper_nn(bits_per_symbol, NUM_NEURONS)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=MeanSquaredError()
    )

    # 6) Train
    batch_size = args.batch_size or bits_per_symbol
    print(f"\nTraining {SCHEME} mapper on {X.shape[0]} symbols (mu={mu})")
    history = model.fit(
        X, Y,
        epochs=args.epochs,
        batch_size=batch_size,
        verbose=2
    )

    # 7) Save the model to "mapper_nn_<SCHEME>.h5"
    out_filename = f"mapper_nn_{SCHEME}.h5"
    model.save(out_filename)
    print(f"\nSaved trained mapper to {out_filename}")

    # 8) Plot training loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{SCHEME} Mapper Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
