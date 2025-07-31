#####Jack Rawls
####jackrawlscollege@gmail.com
##fzb297@mocs.utc.edu
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import layers, models, Input, Dense, Conv1D, Flatten, Reshape, concatenate
###Testing the quantization network first
########M bits are message bits like 16 bit binary value
#K bits are the key bits where num of bits and key used for encription (also in binary) 
#Key-message systems: When you have both a message and a key, the key affects the way the message is encoded, transmitted, or quantized. The model has to learn how both pieces of information should be processed together to produce the final quantized output (or ciphertext).
class SimpleQuantization(tf.keras.layers.Layer):
 def __init__(self,num_levels=16):
  #Calling the constructor of parent class tf.keras.layers.layer
  super(SimpleQuantization,self).__init__()
  self.num_levels= num_levels #set the number of quantization levels
  self.scale = tf.Variable(1.0) ##The trainable scaling factor
 

 def call(self,inputs): 
  #Scaling the input by the learned scaling factor

scaled_input = inputs * self.scale


#Quantizing the input by doing rounding to the nearest quantization level
quantized_output = tf.round(scaled_input * (self.num_levels-1))/(self.num_levels-1)

return quantized_output

##This class represents the neurual network which takes meesgae and key inputs,
##Processes them through layers and applies quantization at the end
class Alice:

 def __init__(self,m_bits,k_bits,num_levels=16):


#Initialize the model with the message m bits and the key k bits bit lengths
#Define the input layers for the message and key bits
self.in1= Input(shape=(m_bits,))#Input for message m bits
self.in2 = Input(shape=(k_bits,))#Input for the key bits k bits
self.num_levels = num_levels #Number of quantization levels
##Concatenating m bits and k bits into 1 vector basically
self.input= concatenate([self.in1,self.in2],axis=1)

##Building the model 
##Defines the architecture of the NN using DENSE and Conv1d layers. 
#It processes the concatenedated inputs and extracts features and quantizes the output


#######################Dense layer is a fully connected layer, meaning each neuron in this layer is connected to every neuron in the previous layer. The Dense layer processes the input data and produces output by learning a weighted sum of the inputs.
#Units = This parameter specifies the number of neurons in the Dense layer. In your case, the number of neurons is equal to the combined size of self.in1 (message input) and self.in2 (key input). So, self.in1.shape[1] + self.in2.shape[1] gives you the number of neurons in the Dense layer, which corresponds to the combined input size (message bits + key bits).
#Relu - applied to the output of each neuron. ReLU introduces non-linearity into the model, meaning it allows the network to learn more complex patterns than if the output were simply a weighted sum. ReLU outputs 0 for negative values and the input value itself for positive values.
#purpose - The Dense layer here is used to process the combined message and key inputs. It learns a new representation of the data, which will then be passed through subsequent layers.
#Reshape- The Reshape layer is used to change the shape of the data so it can be processed by the next layer, which in this case is a Conv1D layer. The Conv1D layer expects its input to be in a specific format (3D tensor: batch_size, sequence_length, num_channels).
#-1 - automatically adjusts the first dimension(batch sizw)to fit the total number of elemetns
#1- this adds the thrid dimension with size 1, essentially it is adding a channel dimension to the input data
#Conv1d layers expect a 3d tensor input, so we need to reshape the output of the dense layer (2d tensor) into  a 3d tensor to pass it to conv1d layers.
#
#conv1d layer - This layer applies a 1D convolution to the input. Convolutional layers are great for learning local patterns in sequential data, like time-series or sequences (in this case, your message + key data).
#filters =2 - The number of filters (kernels) used in the convolution. Each filter is a small weight matrix that will be learned during training. Each filter detects different features in the input sequence. Here, you're using 2 filters to learn 2 different features from the sequence.
#Kenrel size = 3 - The sixe of the filter (kernel). this means each filter will look at 3 consecutive eleents from the input sequence at a time
#Strides = 1 - This means the filtwr will move 1 step at a time, lookingat every possible sequence of 3 consecutive values
#Padding = same - This ensures the output of the Conv1D layer has the same length as the input sequence, even after applying the convolution. Padding adds extra values to the input at the borders to keep the sequence length unchanged.
#Conv1d = This layer detects local patterns in the input sequence (message + key). For example, it could learn basic features like small shifts or trends in the sequence of bits.
#
#2nd conv1d layer- 
#This is another convolutional layer, but it uses 4 filters and a smaller kernel size of 2. The purpose is to learn more abstract or complex patterns from the output of the first convolutional layer.
#         The second convolutional layer helps the model learn more complex patterns in the data after the first layer has already detected some basic features.
#
##Flatten layer - This layer takes the output of the previous layers (which is in a 3D tensor shape) and flattens it into a 1D tensor (i.e., a vector). This is necessary because the next layer, which is typically a Dense layer, expects a 1D input.
#The Flatten layer converts the multi-dimensional data into a single long vector, making it compatible with fully connected Dense layers.
#We then do simpleQuantization which takes ocntinous values and converts them into discrete values
#The output is quantized into discrete levels, typically for compression or for mapping continuous values to a smaller range. The num_levels defines how many discrete levels the data will be mapped to.
#tf.keras.model()- his creates the final model. It takes the input layers (self.in1 and self.in2, which represent the message and key) and the final output (x, which is the quantized output).
#This returns the full model, which can then be compiled, trained, and used for making predictions.
# 
def build_model(self):
 #Dense layer processes the combined input
 x= Dense(units=(self.in1.shape[1] + self.in2.shape[1]), activation  = 'relu')(self.input)
 #Reshape output to prepare it for COnv1d layers and the conv1d requires a 3d input
 x= Reshape((-1,1))(x)##Converting it to  3D Tensor format: (batch_size,new_dim,1)
#Conv1d layer number 1 learns the local patters from this sequence of data
x= Conv1D(filters=2,kernel_size=3,strides=1,activation='relu',padding='same')(x)

##The second conv1d layer Learns the additional local patters
x= Conv1D(filters=4,kernel_size=2,strides=1,activation='relu',padding='same')(x)
##Flatten the three dimensional output to 1 dimension so it can be processed by dense layers
#Dense layers only process 1 d
#After the Conv1D layers, the shape of x is still 3D
#But the next layer that processes this data is the SimpleQuantization layer, which expects a 1D vector as input (it doesn’t understand the 3D shape). This is where Flatten comes in.
x=Flatten()(x)
##Quantiing the output using the simple_quantixation method defined earlier
x= SimpleQuantization(num_levels=self.num_levels)(x)
#Return the model
#tf.keras.Model(): This creates the final model. It takes the input layers (self.in1 and self.in2, which represent the message and key) and the final output (x, which is the quantized output).
#This returns the full model, which can then be compiled, trained, and used for making predictions.
return tf.keras.Model([self.in1,self.in2],x,name='alice')

if __name__ == "__main__":
 #Set up Model parameters
 m_bits=16 #message bit length
 k_bits = 16 #Key bit length
 num_levels=16 #Number of quantization levels

 ###Creating a instance of the Alice class and building the model 
alice_model = Alice(m_bits,k_bits, num_levels).build_model()
#At this point, alice_model is a fully built neural network ready for training.


#Compile the model with Adam optimizer and Mean squared error loss function
#The compile method configures the model for training. It specifies the optimizer and the loss funciton that the model will use during training
#Optimizer adjusts the <1 or >1 is traning scale factor
#Loss function - The Mean Squared Error (MSE) loss function is used for regression tasks where the model predicts continuous values. The MSE loss computes the average of the squared differences between the predicted values and the actual target values. It is commonly used in tasks like quantization, where the goal is to minimize the difference between the predicted quantized output and the true values.
#Purpose of compile: he compile step is important because it prepares the model by defining how it will learn from the data during training. The optimizer is responsible for adjusting the model's parameters (e.g., weights), and the loss function measures how well the model's predictions match the actual values.

alice_model.compile(optimizer ='adam',loss='mse')

#Creating dummy training data(Random data for testing)
#Generates random integers betwen 0 and 1 which simulates binary data, 100 samples are used and each sample is a vector length of m_bits 
x1=np.random.randint(0,2, (100, m_bits))
#Dummy key data of 0s and 1s of n100 samples of key bits
x2=np.random.randint(0,2, (100, k_bits))
#Target output data for training, in a real world scenario this would be the actual encoded or quantized meesage
y=np.random.randint(0,2, (100,m_bits)) 

#Traning the model with the dummy data 
#(input, expected output, epochs)
alice_model.fit([x1,x2],y,epochs=10)

#After training using the model to make predictions
predictions= alice_model.predict([x1,x2])

##Print the predictions for the first 5 samples
print(predictions[:5])  

 