##Neural Network for the mapper block
#Fzb297@mocs.utc.edu
from tensorflow.keras.layers import Input, Dense, Reshape, TimeDistributed 
from tensorflow.keras.models import Model   

##Function to define the neural network for the mapper block 
def build_mapper_nn(bits_per_symbol, num_neurons):
    #Maps a vector of bits per symbol to 2 dimensional i, q vector pair
    #Returns a keras model that outputs i, q
    #input for the symbols number of bits
    inp= Input(shape=(bits_per_symbol,),name='numbits_for_symbol')
    #dense layers which basically understand the data and uses mathematicl equations to add a weights and bias vectors 
    x=Dense(num_neurons, activation='relu', name='mapper_dense1')(inp)
    x=Dense(num_neurons,activation ='relu', name='mapper_dense2')(x)
    out= Dense(2, name='mapper_IQ')(x) 
    return Model(inputs= inp,outputs=out, name='MapperNN') ##Returning the NN model

##Building the full mapper after we built the neural network 
##Need to break all bits in ofdm symbol intoi 64 separate group of 4 bits

def build_full_mapper(num_subcarriers, bits_per_symbol, num_neurons):
    ##Holds a ofdm symbols entire bit stream like all bits going parallel across parallel lanes
    ##Example 64 subcarriers or parallel lanes * 4 bits per symbol is a legnth of 256 which is bits per ofdm symbols
    serial_in=Input(shape=(num_subcarriers * bits_per_symbol,), name='serial_bits')
    ##One ofdm symbol is what this input means 
    #Bitstream is still 1 d
    #needs to be 2d, because the paralell lanes are 2d one row per subcarrier
    p=Reshape((num_subcarriers,bits_per_symbol),name='reshape_for_mapper')(serial_in)
    #P has shape (num subcars, bits per smbol) each row is the bits for one subcarrier

    #call mapper nn
    mapper_nn=build_mapper_nn(bits_per_symbol, num_neurons)
    #Outputs i and q 
    ##Apply mapper to every subcarrier in parallel
   #Instead of using a loop we just use timedistributed
    mapped= TimeDistributed(mapper_nn, name='map_each_symbol')(p)
    return Model(inputs=serial_in,outputs=mapped,name='FullMapper')








	









