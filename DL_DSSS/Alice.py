import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
from DL_DSSS import Models
from DL_DSSS.Loss import Loss
from DL_DSSS.Datagen import Data
from tensorflow.keras.optimizers import Adam, RMSprop
from DSSS_Train import NN_setup, framework_train
from tensorflow.keras.models import Model
import numpy as np


#tf.keras.utils.get_custom_objects().update({'OneHotLayer': Models.OneHotLayer})
def Alice_Load_Quantize(model_file, n_samples, num_bits):
    Alice_out, Alice_channel, tst_msg, tst_key = Alice_output(model_file, n_samples)
    # print("Shape Alice_out", Alice_out.shape)
    # print("Shape Alice_channel", Alice_channel.shape)
    # print("Alice_channel", Alice_channel)

    Alice_out_quantized = []
    for vector in Alice_out:
        quantized_vector = quantize_float_to_bits(vector, num_bits)
        Alice_out_quantized.append("".join(quantized_vector))
    Alice_out_stream = "".join(Alice_out_quantized)
    # print("Length Alice out stream: ", len(Alice_out_stream))
    with open('Alice_output_stream.bin', 'w') as file:
        file.write(Alice_out_stream)
    #np.savetxt('Alice_out_test_messages', Alice_out_stream, fmt='%d')
    return np.array(Alice_out_quantized), Alice_out_stream, Alice_out, Alice_channel, tst_msg, tst_key

def Alice_output(model_file, n_samples):
    abmodel = tf.keras.models.load_model(model_file)
    bits = Data(16, 15)
    tst_msg, tst_key = bits.create_test_data(n_samples)

    # Extract Alice from abmodel
    alice_out = abmodel.get_layer('alice').output
    alice_in = abmodel.get_layer('alice').input
    alice_model = Model(inputs=alice_in, outputs=alice_out, name='ALICE')
    # alice_model = tf.keras.models.load_model(model_file)
    predictions = alice_model.predict([tst_msg, tst_key])

    A_out, Achannel_out = predictions[0], predictions[1]
    # print("Aout:", A_out[0])
    # print("Achannel_out:", Achannel_out[0])

    return A_out, Achannel_out, tst_msg, tst_key

def quantize_float_to_bits(vector, num_bits):
    vector_out = []
    for value in vector:
        # Ensure the value is within the valid range (0, 1)
        value = max(0.0, min(1.0, value))

        # Convert the value to a quantized integer in the range [0, 2^num_bits - 1]
        quantized_value = int(value * (2 ** num_bits - 1))

        # Convert the integer to a binary string representation with leading zeros
        binary_representation = format(quantized_value, f'0{num_bits}b')
        vector_out.append(binary_representation)

    return vector_out


def dequantize_bits_to_float(bits, num_bits):
    # Convert the binary string to an integer
    quantized_value = int(bits, 2)

    # Convert the integer back to the original float value
    value = quantized_value / (2 ** num_bits - 1)

    return value


if __name__ == '__main__':
    # Test the quantization and dequantization
    original_float = 0.75
    num_bits = 4

    quantized_bits = quantize_float_to_bits([original_float], num_bits)
    print(f"Original float: {original_float}")
    print(f"Quantized bits: {quantized_bits}")

    dequantized_float = [dequantize_bits_to_float(bit, num_bits) for bit in quantized_bits]
    print(f"Dequantized float: {dequantized_float}")