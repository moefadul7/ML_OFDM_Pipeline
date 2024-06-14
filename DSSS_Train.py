import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
from DL_DSSS import Models
from DL_DSSS.Loss import Loss
from DL_DSSS.Datagen import Data
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import Callback

class LossThresholdCallback(tfk.callbacks.Callback):
    def __init__(self, loss_limit):
        super(LossThresholdCallback, self).__init__()
        self.loss_limit = loss_limit

    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') < self.loss_limit:
            print(f"\nStopping training as loss reached below {self.loss_limit}")
            self.model.stop_training = True

def NN_setup(m_bits, k_bits, c_bits):
    # Create NN for Alice, Bob, & Eve

    alice = Models.Alice(m_bits, k_bits, c_bits)
    Alice_Model = alice.build_model()
    # print(Alice_Model.summary())

    bob = Models.Bob(m_bits, k_bits, c_bits)
    Bob_Model = bob.build_model()
    # print(Bob_Model.summary())

    eve = Models.Eve(m_bits, k_bits, c_bits)
    Eve_Model = eve.build_model()
    # print(Eve_Model.summary())

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
    # print("bob_concatenated_output", bob_concatenated_output)

    # print("concatenated input: ", concatenated_input)
    # print("eve_concatenate_output: ", eve_concatenated_output)
    # Loss place holders:

    eve_loss = Loss(concatenated_input, eve_concatenated_output)
    bob_loss = Loss(concatenated_input, bob_concatenated_output)

    #abeloss = Loss(alice.in1, alice_out, bob_out, eve_out)
    abeloss = bob_loss.loss + tf.math.square(m_bits / 2 - eve_loss.loss) / ((m_bits // 2) ** 2)

    # Learning optimizers:
    abeoptim = RMSprop(learning_rate=0.001)
    eveoptim = RMSprop(learning_rate=0.001)

    # Create Macro models for alice-bib & alice-eve
    abmodel = Models.Macro([alice.in1, alice.in2], bob_out, 'abmodel', abeloss, abeoptim)
    abmodel.compile()
    Alice_Model.trainable = False

    evemodel = Models.Macro([alice.in1, alice.in2], eve_out, 'evemodel', eve_loss.loss, eveoptim)
    evemodel.compile()

    return Alice_Model, Bob_Model, Eve_Model, abmodel, evemodel


def framework_train(m_bits, k_bits, n_batches, batch_size, n_epochs, Alice_Model, Bob_Model, abmodel, evemodel,
                    abecycles, evecycles, n_samples, model_file):
    ## Get Messages and Codes
    data = Data(m_bits, k_bits)
    Messages = data.train_messages
    Codes = data.train_codes
    loss_callback = LossThresholdCallback(loss_limit=0.1)

    ## Initialize training loop:
    epoch = 0
    Bob_Err = []
    Eve_Err = []
    while epoch < n_epochs:
        for iteration in range(n_batches):
            Alice_Model.trainable = True
            m_batch = Messages[iteration * batch_size: (iteration + 1) * batch_size]
            k_batch = Codes[iteration * batch_size: (iteration + 1) * batch_size]

            # Train Alice-Bob
            for cycle in range(abecycles):
                historyA = abmodel.model.fit([m_batch, k_batch], None, epochs=1, verbose=1, callbacks=[loss_callback])
                print("Epoch {}, Iteration {}. Alice-Bob Loss is: {} \n".format(epoch, iteration, historyA.history['loss'][-1]))

            Alice_Model.trainable = True
            for cycle in range(evecycles):
                historyE = evemodel.model.fit([m_batch, k_batch], None, epochs=1, verbose=1, callbacks=[loss_callback])
                print("Epoch {}, Iteration {}. Eve Loss is: {} \n".format(epoch, iteration, historyE.history['loss'][-1]))

        Bob_Err.append(historyA.history['loss'][-1])
        Eve_Err.append(historyE.history['loss'][-1])

        epoch += 1
        if abmodel.model.stop_training or evemodel.model.stop_training:
            break

    # Model Evaluation:
    # generate test data
    msg_tst, code_tst = data.create_test_data(n_samples)

    # model prediction
    Bob_out = abmodel.model.predict([msg_tst, code_tst])
    # print("Bob _out Train: ", Bob_out[0])
    # print("Bob_out shape Trained: ", Bob_out.shape)
    Ber = Loss(msg_tst, Bob_out[0])

    # Display evaluation results:
    print('Avg Ber of the test set: \n', Ber.loss.numpy())

    # Save the trained abmodel
    print("saving trained models ... \n")
    abmodel.model.save(model_file)
    # Alice_Model.save('Alice.keras')
    # Bob_Model.save('Bob.keras')

    print("Generating training evolution results ... \n")

    # Plot training results
    fig, ax = plt.subplots()
    ax.plot(Bob_Err, 'k-', label='Bob Error')
    ax.plot(Eve_Err, 'r-', label='Eve Error')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Bit Error')
    plt.savefig('training_evolution.png')
    plt.show()
    # print(Bob_Err)
    # print(Eve_Err)


