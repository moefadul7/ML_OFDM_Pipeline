import tensorflow as tf


class Loss:

    def __init__(self, inputs, output):
        if isinstance(inputs, list):
            self.input = tf.concat(inputs,axis=-1)
        else:
            self.input = inputs
        self.output = output
        self.loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.abs(self.input - self.output), axis=-1))



"""
class Loss:

    def __init__(self, input_message, input_channel, output_message, output_channel):
        self.input_message = input_message
        self.input_channel = input_channel
        self.output_message = output_message
        self.output_channel = output_channel

        self.message_loss = tf.math.reduce_mean(
            tf.math.reduce_sum(tf.math.abs(self.input_message - self.output_message), axis=-1))
        self.channel_loss = tf.math.reduce_mean(
            tf.math.reduce_sum(tf.math.abs(self.input_channel - self.output_channel), axis=-1))

        self.loss = self.message_loss + self.channel_loss
"""