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

class Alice:
    def __init__(self, m_bits, k_bits, c_bits):
        self.in1 = Input(shape=(m_bits,)) #message
        self.in2 = Input(shape=(k_bits,))
        self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits

    def build_model(self):
        adense1 = Dense(units=(self.m_bits + self.k_bits), activation=tf.nn.tanh)(self.input)
        areshape = Reshape((self.m_bits + self.k_bits, 1,))(adense1)
        aconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh, padding=pad)(areshape)
        aconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh, padding=pad)(aconv1)
        aconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh, padding=pad)(aconv2)
        aconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid, padding=pad)(aconv3)
        aoutput = Flatten()(aconv4)

        return Model([self.in1, self.in2], aoutput, name='alice')

class Bob:
    def __init__(self, m_bits, k_bits, c_bits):
        self.in1 = Input(shape=(c_bits,)) #message
        self.in2 = Input(shape=(k_bits,))
        self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits
        self.k_bits = k_bits
        self.c_bits = c_bits

    def build_model(self):
        bdense1 = Dense(units=(self.c_bits + self.k_bits), activation=tf.nn.tanh)(self.input) # change here
        breshape = Reshape((self.c_bits + self.k_bits, 1,))(bdense1)
        bconv1 = Conv1D(filters=2, kernel_size=4, strides=1, activation=tf.nn.tanh ,padding=pad)(breshape)
        bconv2 = Conv1D(filters=4, kernel_size=2, strides=2, activation=tf.nn.tanh ,padding=pad)(bconv1)
        bconv3 = Conv1D(filters=4, kernel_size=1, strides=1, activation=tf.nn.tanh ,padding=pad)(bconv2)
        bconv4 = Conv1D(filters=1, kernel_size=1, strides=1, activation=tf.nn.sigmoid ,padding=pad)(bconv3)
        boutput = Flatten()(bconv4)

        return Model([self.in1, self.in2], boutput, name='bob')

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

        return Model(self.input, eoutput, name='eve')

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
    A = Alice(m_bits, k_bits, c_bits)
    B = Bob(m_bits, k_bits, c_bits)
    E = Eve(m_bits, k_bits, c_bits)
    print("Below is the summary for Alice's model: \n")
    print(A.build_model().summary())
    print("Below is the summary for Bob's model: \n")
    print(B.build_model().summary())
    print("Below is the summary for Eve's model: \n")
    print(E.build_model().summary())
    
    
# Main 

if __name__ == "__main__":
    main()
