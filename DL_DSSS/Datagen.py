import itertools
import numpy as np

class Data:
    def __init__(self, m_bits, k_bits):
        self.m_bits = m_bits
        self.k_bits = k_bits
        msg_lst = list(itertools.product([0, 1], repeat=m_bits))
        key_lst = list(itertools.product([0, 1], repeat=k_bits))

        key_lst = np.array(key_lst)
        self.train_messages = np.array(msg_lst)
        self.train_codes = np.concatenate((key_lst, key_lst), axis=0)

        np.random.shuffle(self.train_messages)
        np.random.shuffle(self.train_codes)


    def create_test_data(self, n_samples):
        idxs = np.random.randint(0, 2**self.m_bits, n_samples)
        msg_tst = self.train_messages[idxs]
        code_tst = self.train_codes[idxs]

        # writing test data to files

        np.savetxt('test_messages', msg_tst, fmt='%d')
        np.savetxt('test_codes', code_tst, fmt='%d')

        # with open("test_meaages.txt", "w") as m_file:
        #     m_file.write(msg_tst)
        # with open("test_codes.txt", "w") as c_file:
        #     c_file.write(code_tst)
        return msg_tst, code_tst
