import tensorflow as tf
import numpy as np
from pilotnn import build_pilot
##for testing
m_bits=16
k_bits=16
p=8
pilot_model= build_pilot(m_bits,k_bits,p)
#prints a table showing each layers name and type, output shape and number of trainable paramters in that layer
pilot_model.summary()
#Dummy data via numpy
##input data for pilotnn. shape = (1,32) (1,mbits +kbits
x= np.random.randint(0,2,size=(1,m_bits+k_bits)).astype(float)
#Output of pilotnn
#shape = (1,2*P)(1,16) for p=8 #first 8 are real parts and last 8 are imaginary
y=pilot_model.predict(x) ##forward pass this is the output
#.predict is the output
print("y.shape:", y.shape) #prints (1,16)  