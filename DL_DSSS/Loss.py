import tensorflow as tf

class Loss:
    def __init__(self,input, output):
        self.input = input
        self.output = output
        self.loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.abs(input - output), axis=-1))



