import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def Alice_Load_Quantize(model_file, n_samples, num_bits):
    # Load pre-trained Alice model
    nn_model = load_model(model_file) 

    # Generate random test data
    tst_msg = np.random.randint(0, 2, (n_samples, 16))  # 16-bit messages
    tst_key = np.random.randint(0, 2, (n_samples, 16))  # 15-bit keys for testing maybe changing to 15

    #  Pad tst_key to 16 bits before prediction
    if tst_key.shape[1] < 16:
        print("Padding tst_key from 15 to 16 bits")
        tst_key = np.pad(tst_key, ((0, 0), (0, 1)), mode='constant')

    # Predict binary output from neural network
    Alice_out_binary = (nn_model.predict([tst_msg, tst_key]) > 0.5).astype(int)

    # Convert to bitstream string (flattened 0s and 1s)
    Alice_out_stream = "".join(str(int(bit)) for bit in Alice_out_binary.flatten())

    # Save to file (Bob will read this)
    with open('Alice_output_stream', 'w') as file:
        file.write(Alice_out_stream)

    return Alice_out_binary, Alice_out_stream, Alice_out_binary, tst_msg, tst_key

# For testing
if __name__ == "__main__":
    quantized, stream, _, msg, key = Alice_Load_Quantize("alice_quantization_model.h5", 10, 8)
    print("Quantized output shape:", quantized.shape)
    print("First 100 bits of stream:", stream[:100])
