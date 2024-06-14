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
from tensorflow.keras.layers import Lambda
from Loss import Loss

# Vars
pad = 'same'


def add_channel_selection_layer(model):
    original_output = model.output
    # print(original_output)
    channel_output = Dense(35, activation='softmax')(original_output)
    # print(channel_output)
    new_model = Model(inputs=model.input, outputs=[original_output, channel_output])
    return new_model


class Alice:
    def __init__(self, m_bits, k_bits, c_bits, s_bits):
        self.in1 = Input(shape=(m_bits,))  # message
        self.in2 = Input(shape=(k_bits,))
        self.in3 = Input(shape=(s_bits,))
        self.input = concatenate([self.in1, self.in2, self.in3], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits
        self.s_bits = s_bits

    def build_model(self):
        combined_bits = self.m_bits + self.k_bits + self.s_bits
        adense1 = Dense(units=(combined_bits), activation=tf.nn.tanh)(self.input)
        # adense1 = Dense(units=(self.m_bits + self.k_bits), activation=tf.nn.tanh)(self.input)
        areshape = Reshape((combined_bits, 1,))(adense1)
        # areshape = Reshape((self.m_bits + self.k_bits))
        aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh, padding=pad)(areshape)
        aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh, padding=pad)(aconv1)
        aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh, padding=pad)(aconv2)
        aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid, padding=pad)(aconv3)
        aoutput = Flatten()(aconv4)

        amodel = Model([self.in1, self.in2, self.in3], aoutput, name='alice')
        final_model = add_channel_selection_layer(amodel)
        return final_model


class Bob:
    def __init__(self, m_bits, k_bits, c_bits, s_bits):
        self.in1 = Input(shape=(c_bits,))  # message
        self.in2 = Input(shape=(k_bits,))
        self.in3 = Input(shape=(s_bits,))
        self.input = concatenate([self.in1, self.in2, self.in3], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits
        self.s_bits = s_bits

    def build_model(self):
        combined_bits = self.c_bits + self.k_bits + self.s_bits
        bdense1 = Dense(units=(combined_bits), activation=tf.nn.tanh)(self.input)  # change here
        breshape = Reshape((combined_bits, 1,))(bdense1)
        bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh, padding=pad)(breshape)
        bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh, padding=pad)(bconv1)
        bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh, padding=pad)(bconv2)
        bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid, padding=pad)(bconv3)
        boutput = Flatten()(bconv4)

        model = Model([self.in1, self.in2, self.in3], boutput, name='bob')
        return add_channel_selection_layer(model)


class Eve:
    def __init__(self, m_bits, k_bits, c_bits):
        self.input = Input(shape=(c_bits,))  # message
        # self.in2 = Input(shape=(k_bits,))
        # self.input = concatenate([self.in1, self.in2], axis=1)
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
        eoutput = Flatten()(econv4)  # Eve's attempt at guessing the plaintext

        model = Model(self.input, eoutput, name='eve')
        return add_channel_selection_layer(model)


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
    s_bits = 5


    alice = Alice(m_bits, k_bits, c_bits, s_bits)
    Alice_Model = alice.build_model()
    print(Alice_Model.summary())

    bob = Bob(m_bits, k_bits, c_bits, s_bits)
    Bob_Model = bob.build_model()
    print(Bob_Model.summary())

    eve = Eve(m_bits, k_bits, c_bits)
    Eve_Model = eve.build_model()
    print(Eve_Model.summary())

    # Generate outputs of each model:
    alice_out = Alice_Model([alice.in1, alice.in2])
    alice_message, alice_channel = alice_out


    bob_out = Bob_Model([alice_message, alice.in2])
    bob_message, bob_channel = bob_out


    eve_out = Eve_Model(alice_message)
    eve_message, eve_channel = eve_out

    concatenated_input = Lambda(lambda x: tf.concat(x, axis=-1))([alice.in1, alice_channel])
    bob_concatenated_output = Lambda(lambda x: tf.concat(x, axis=-1))([bob_message, bob_channel])
    eve_concatenated_output = Lambda(lambda x: tf.concat(x, axis=-1))([eve_message, eve_channel])
    print("bob_concatenated_output", bob_concatenated_output)

    print("concatenated input: ", concatenated_input)
    print("eve_concatenate_output: ", eve_concatenated_output)
    # Loss place holders:
    eve_loss = Loss(concatenated_input, eve_concatenated_output)
    bob_loss = Loss(concatenated_input, bob_concatenated_output)

    # abeloss = Loss(alice.in1, alice_out, bob_out, eve_out)
    abeloss = bob_loss.loss + tf.math.square(m_bits / 2 - eve_loss.loss) / ((m_bits // 2) ** 2)

    # Learning optimizers:
    abeoptim = RMSprop(learning_rate=0.001)
    eveoptim = RMSprop(learning_rate=0.001)

    # Create Macro models for alice-bib & alice-eve
    abmodel = Macro([alice.in1, alice.in2, alice.in3], bob_out, 'abmodel', abeloss, abeoptim)
    abmodel.compile()
    print("abmodel summary", abmodel.model.summary())
    Alice_Model.trainable = False

    evemodel = Macro([alice.in1, alice.in2, alice.in3], eve_out, 'evemodel', eve_loss.loss, eveoptim)
    evemodel.compile()
    print("evemodel summary", evemodel.model.summary())



    input_A = [np.random.randn(1, m_bits), np.random.randn(1, k_bits), np.random.randn(1, s_bits)]
    input_B = [np.random.randn(1, c_bits), np.random.randn(1, k_bits), np.random.randn(1, s_bits)]
    input_E = np.random.randn(1, c_bits)

    output_A = Alice_Model.predict(input_A)
    output_B = Bob_Model.predict(input_B)
    output_E = Eve_Model.predict(input_E)

    def softmax_to_one_hot(softmax_output):
        one_hot_output = np.zeros_like(softmax_output)
        one_hot_output[np.arange(len(softmax_output)), softmax_output.argmax(axis=1)] = 1
        return one_hot_output

    # Print the outputs for Alice, Bob, and Eve

    print("Alice's Outputs:")
    print("Communication Data:", output_A[0])
    print("Channel Selection (binary):", softmax_to_one_hot(output_A[1]))

    # print(Alice.in1)

    print("\nBob's Outputs:")
    print("Bob out", output_B)
    print("Communication Data:", output_B[0])
    print("Channel Selection (binary):", softmax_to_one_hot(output_B[1]))

    print("\nEve's Outputs:")
    print("Eve out: ", output_E)
    print("Communication Data:", output_E[0])
    print("Channel Selection (binary):", softmax_to_one_hot(output_E[1]))

    """
    print("Below is the summary for Alice's model: \n")
    print(A.build_model().summary())
    print("Below is the summary for Bob's model: \n")
    print(B.build_model().summary())
    print("Below is the summary for Eve's model: \n")
    print(E.build_model().summary())
    """


# Main

if __name__ == "__main__":
    main()