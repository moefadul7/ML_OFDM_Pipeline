# Load bit from file
import numpy as np

bit_list = []
bit_stream = ''



def bit_stream_loader(file_name):
    bit_list = []
    bit_stream = ''
    # with open(file_name) as f:
    #     lines = [z.replace('\n', '') for z in f.readlines()]
    #     for x in lines:
    #         line = []
    #         y = x.split(' ')
    #         digits = [int(digit) for digit in y]
    #         bit_list.append(digits)
    # bit_stream  = " ".join(lines)
    with open(file_name) as f:
        bit_stream = f.readlines()
    #Tx_stream = bit_stream.replace(" ", "")
    #Tx_bit_stream = np.array([int(bit) for bit in bit_stream])
    Tx_bit_stream = np.array([int(bit) for bit in bit_stream[0]])
    #return Tx_bit_stream, bit_stream, np.array(bit_list)
    return Tx_bit_stream, bit_stream

def Dump_Rcvd_stream(file_name, Rcv_bit_stream, span):
    dump = ''
    for i in range(span):
        dump += str(Rcv_bit_stream[i])

    with open(file_name, 'w') as f:
        f.write(dump)

if __name__ == '__main__':
    filename = 'test_messages'
    bit_stream, bit_lst = bit_stream_loader(filename)
    print('bit-list shape: {}'.format(np.array(bit_lst).shape))
    print('bit-stream shape: {}'.format(len(bit_stream)))

