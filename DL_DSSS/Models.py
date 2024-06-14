# This script define Alice, Bob, & Eve model objects
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as tfk
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.layers import Flatten, Reshape, concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Vars
pad = 'same'

"""
class OneHotLayer(Layer):
    def call(self, inputs):
        one_hot = tf.one_hot(tf.argmax(inputs, axis=-1), depth=inputs.shape[-1])
        return one_hot
def add_channel_selection_layer(model):
    original_output = model.output
    channel_layer = Dense(35)(original_output)
    channel_output = OneHotLayer()(channel_layer)
    new_model = Model(inputs=model.input, outputs=[original_output, channel_output], name=model.name)
    return new_model
"""
def add_channel_selection_layer(model):
    original_output = model.output
    #print(original_output)
    channel_output = Dense(35, activation='softmax')(original_output)
    #print(channel_output)
    new_model = Model(inputs=model.input, outputs=[original_output, channel_output], name = model.name)
    return new_model

class Alice:
    def __init__(self, m_bits, k_bits, c_bits):
        self.in1 = Input(shape=(m_bits,)) #message
        self.in2 = Input(shape=(k_bits,))

        self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits

    def build_model(self):
        combined_bits = self.m_bits + self.k_bits
        adense1 = Dense(units=(combined_bits), activation=tf.nn.tanh)(self.input)
        #adense1 = Dense(units=(self.m_bits + self.k_bits), activation=tf.nn.tanh)(self.input)
        areshape = Reshape((combined_bits, 1,))(adense1)
        #areshape = Reshape((self.m_bits + self.k_bits))
        aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh, padding=pad)(areshape)
        aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh, padding=pad)(aconv1)
        aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh, padding=pad)(aconv2)
        aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid, padding=pad)(aconv3)
        aoutput = Flatten()(aconv4)

        amodel = Model([self.in1, self.in2], aoutput, name='alice')
        return add_channel_selection_layer(amodel)

class Bob:
    def __init__(self, m_bits, k_bits, c_bits):
        self.in1 = Input(shape=(c_bits,)) #message
        self.in2 = Input(shape=(k_bits,))

        self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits


    def build_model(self):
        combined_bits = self.c_bits + self.k_bits
        bdense1 = Dense(units=(combined_bits), activation=tf.nn.tanh)(self.input) # change here
        breshape = Reshape((combined_bits, 1,))(bdense1)
        bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh ,padding=pad)(breshape)
        bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh ,padding=pad)(bconv1)
        bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh ,padding=pad)(bconv2)
        bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid ,padding=pad)(bconv3)
        boutput = Flatten()(bconv4)

        bmodel = Model([self.in1, self.in2], boutput, name='bob')
        return add_channel_selection_layer(bmodel)

class Eve:
    def __init__(self, m_bits, k_bits, c_bits):
        self.input = Input(shape=(c_bits,))  # message
        #self.in2 = Input(shape=(k_bits,))
        #self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits


    def build_model(self):
        edense1 = Dense(units=self.c_bits, activation=tf.nn.tanh)(self.input)
        edense2 = Dense(units=(self.c_bits + self.k_bits), activation=tf.nn.tanh)(edense1)
        ereshape = Reshape((self.c_bits + self.k_bits, 1,))(edense2)
        econv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh, padding=pad)(ereshape)
        econv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh, padding=pad)(econv1)
        econv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh, padding=pad)(econv2)
        econv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid, padding=pad)(econv3)
        eoutput = Flatten()(econv4)# Eve's attempt at guessing the plaintext

        emodel = Model(self.input, eoutput, name='eve')
        return add_channel_selection_layer(emodel)

class Macro:
    def __init__(self, inputs, outputs, name, loss, optimizer):
        self.inputs = inputs
        self.outputs = outputs
        self.model = Model(self.inputs, self.outputs, name=name)
        self.loss = loss
        self.optimizer = optimizer

    def compile(self):
        self.model.add_loss(self.loss)
        self.model.compile(optimizer=self.optimizer)

def main():
    m_bits = 16
    k_bits = 15
    c_bits = 16

    alice_model = Alice(m_bits, k_bits, c_bits).build_model()
    bob_model = Bob(m_bits, k_bits, c_bits).build_model()
    eve_model = Eve(m_bits, k_bits, c_bits).build_model()
    """
    # Print the model summaries
    print("Alice Model Summary:")
    alice_model.summary()
    print("\nBob Model Summary:")
    bob_model.summary()
    print("\nEve Model Summary:")
    eve_model.summary()

    # Check and print the names of the models and their outputs
    print("\nChecking model names and output names:")
    print("Alice Model Name:", alice_model.name)
    for layer in alice_model.layers:
        print("Layer name:", layer.name, " - Output name:", layer.output.name)

    """
    input_A = [np.random.randn(1, m_bits), np.random.randn(1, k_bits)]
    input_B = [np.random.randn(1, c_bits), np.random.randn(1, k_bits)]
    input_E = np.random.randn(1, c_bits)

    output_A = alice_model.predict(input_A)
    output_B = bob_model.predict(input_B)
    output_E = eve_model.predict(input_E)



    # Print the outputs for Alice, Bob, and Eve
    
    print("Alice's Outputs:")
    # print("Communication Data:", output_A[0])
    print("Channel Selection (binary):", (output_A[1]))

    print("\nBob's Outputs:")
    # print("Bob out", output_B)
    # print("Communication Data:", output_B[0])
    print("Channel Selection (binary):", (output_B[1]))

    print("\nEve's Outputs:")
    # print("Eve out: ", output_E)
    # print("Communication Data:", output_E[0])
    print("Channel Selection (binary):", (output_E[1]))


# Main
if __name__ == "__main__":
    main()