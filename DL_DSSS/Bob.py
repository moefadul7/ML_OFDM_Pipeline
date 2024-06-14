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
from bit_loader import bit_stream_loader

#tf.keras.utils.get_custom_objects().update({'OneHotLayer': Models.OneHotLayer})


def Bob_Load_Dequantize(model_file, file_name, c_bits, num_bits, tst_key):
    Bob_input = Load_Bob_input(file_name, c_bits, num_bits)
    # print("Bob_Input: ", Bob_input)
    # print("Bob_input shape:", Bob_input.shape)
    Bob_input_Dequantized = []
    for vector in Bob_input:
        floats = dequantize_bits_to_float(vector, num_bits)
        Bob_input_Dequantized.append(floats)
    Bob_input_Dequantized = np.array(Bob_input_Dequantized)

    # Generate Bob's output
    abmodel = tf.keras.models.load_model(model_file)
    bob_out = abmodel.get_layer('bob').output
    bob_in = abmodel.get_layer('bob').input
    bob_model = Model(inputs=bob_in, outputs=bob_out, name='BOB')
    # bob_model = tf.keras.models.load_model(model_file)
    # print("Shape of Bob_input_Dequantized:", Bob_input_Dequantized.shape)
    # print("Shape of tst_key:", tst_key.shape)
    # print('\n printing shapes')
    # print(Bob_input_Dequantized.shape)
    # print(tst_key.shape)
    predictions = bob_model.predict([Bob_input_Dequantized, tst_key])
    B_out, Bchannel_out = predictions[0], predictions[1]
    B_out = np.where(B_out<0.5,0,1)

    return Bob_input_Dequantized, B_out, Bchannel_out


def Load_Bob_input(file_name, c_bits, num_bits):
    input_stream, input = bit_stream_loader(file_name)
    Bob_input = input_stream.reshape(len(input_stream)//(c_bits * num_bits), c_bits, num_bits)
    # print("Bob_input shape", Bob_input.shape)
    return Bob_input



def dequantize_bits_to_float(vector, num_bits):
    vector_out = []
    for value in vector:
        Value = np.array2string(value, separator='').replace('[', '').replace(']', '')

        # Convert the binary string to an integer
        quantized_value = int(Value, 2)

        # Convert the integer back to the original float value
        float_value = quantized_value / (2 ** num_bits - 1)
        vector_out.append(float_value)
    return vector_out

def Bob_Channel_Selection(model_file, tst_key):
    # Dummy input for channel selection
    dummy_input = np.zeros((1, 16))
    abmodel = tf.keras.models.load_model(model_file)
    bob_out = abmodel.get_layer('bob').output
    bob_in = abmodel.get_layer('bob').input
    bob_model = Model(inputs=bob_in, outputs=bob_out, name='BOB')
    predictions = bob_model.predict([dummy_input, tst_key])
    _, Bchannel_out = predictions[0], predictions[1]
    #Bchannel_out = np.where(Bchannel_out < 0.5, 0, 1)

    return Bchannel_out