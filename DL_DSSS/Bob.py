from tensorflow.keras.models import load_model
import numpy as np
from bit_loader import bit_stream_loader

def Load_Bob_input(file_name):
    input_stream, _ = bit_stream_loader(file_name)
    num_samples = len(input_stream) // 16
    input_stream = input_stream[:num_samples * 16]
    return input_stream.reshape(num_samples, 16)

def Bob_Load_Dequantize(model_file, file_name, tst_key):
    print("Bob_Load_Dequantize() is executing...")

    Bob_input = Load_Bob_input(file_name)

    #  DO NOT PAD tst_key – your model expects exactly (None, 15)
    # tst_key should already be (100, 15) from Alice_Load_Quantize

    model = load_model(model_file)

    # Predict
    B_out = model.predict([Bob_input, tst_key])
    B_out = (B_out > 0.5).astype(int)

    print(f"Bob's output shape: {B_out.shape}")
    return Bob_input, B_out
