import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model 
##For now we will leave Snr estimation/backpropgation for that out to determine if the nn is properly ducntioning'
#2 inputs message and key bits and P is the number of pilot bins 
#concatenate into a vector     inp= Input(shape=(bits_per_symbol,),name='numbits_for_symbol')

def build_pilot(m_bits, k_bits,p):
    inp=Input(shape=(m_bits+k_bits,), name='pilot_input')
    x= Dense(64,activation='relu',name='pilot_dense1')(inp)
    x=Dense(32,activation='relu', name='pilot_dense2')(x)
    #2*p is pilot values p is 8 pilot values but you need 8 pilots for real numbers and 8 fr imaginary hence the 2*p
    #YOU HAVE8 BINS BUT have to do 2*8 bc 8 real parts 8 imag parts but in 8 bins wierd at frist but makes sense
    #forms 8 complex pilots. a+bi
    out=Dense(2*p,activation=None, name="pilot_output")(x)
    return Model(inputs=inp,outputs=out,name='PilotNN')
