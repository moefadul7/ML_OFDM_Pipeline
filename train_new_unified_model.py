import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class Alice:
    def __init__(self, m_bits, k_bits):
        self.in1 = Input(shape=(m_bits,))
        self.in2 = Input(shape=(k_bits,))
        self.input = concatenate([self.in1, self.in2], axis=1)
        self.m_bits = m_bits

    def build_model(self):
        dense_units = 128
        x = Dense(dense_units, activation='relu')(self.input)
        x = Dense(dense_units, activation='relu')(x)
        x = Dense(dense_units//2, activation='relu')(x)
        x = Reshape((-1, 1))(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        x = Flatten()(x)
        output = Dense(self.m_bits, activation='sigmoid')(x)
        return tf.keras.Model([self.in1, self.in2], output, name='alice')

# Main Training
if __name__ == "__main__":
    m_bits = 16
    k_bits = 16

    alice_model = Alice(m_bits, k_bits).build_model()
    alice_model.compile(optimizer=Adam(learning_rate=0.001), 
                        loss='binary_crossentropy', metrics=['accuracy'])

    # More training data: 10,000 samples (critical)
    n_samples = 10000
    x1 = np.random.randint(0, 2, (n_samples, m_bits))
    x2 = np.random.randint(0, 2, (n_samples, k_bits))
    y = np.random.randint(0, 2, (n_samples, m_bits))

    # Better Training Settings: 500 epochs
    epochs = 500
    batch_size = 128
    lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, verbose=1)

    history = alice_model.fit(
        [x1, x2], y, epochs=epochs, batch_size=batch_size,
        callbacks=[lr_callback], shuffle=True
    )

    alice_model.save('alice_quantization_model.h5')  

    # Plot training progress
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend() 
    plt.grid(True)
    plt.title('Training Loss and Accuracy')
    plt.savefig('training_performance.png', dpi=300)
    plt.show() 
