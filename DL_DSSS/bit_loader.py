import numpy as np

def bit_stream_loader(file_name, k_bits, m_bits):
    """
    Loads the bit stream from a file and ensures the size is divisible by the symbol length.
    Args:
        file_name (str): The name of the file containing the bit stream.
        k_bits (int): The number of bits for the first symbol.
        m_bits (int): The number of bits for the second symbol.

    Returns:
        np.array: The bit stream loaded from the file, adjusted to the correct size.
    """
    # Load the bit stream from file
    bit_stream = np.loadtxt(file_name, dtype=int)

    # Make sure the length of the bit stream is divisible by the symbol length (k_bits * m_bits)
    required_size = k_bits * m_bits
    remainder = len(bit_stream) % required_size
    if remainder != 0:
        print(f"Warning: The size of input_stream ({len(bit_stream)}) is not divisible by {required_size}. Adjusting the size.")
        bit_stream = bit_stream[:-remainder]  # Truncate excess data

    return bit_stream

def Dump_Rcvd_stream(file_name, Rcv_bit_stream, span):
    dump = ''
    for i in range(span):
        dump += str(Rcv_bit_stream[i])

    with open(file_name, 'w') as f:
        f.write(dump)

if __name__ == '__main__':
    filename = 'test_messages'
    
    # You should define k_bits and m_bits according to your system
    k_bits = 8  # Example value
    m_bits = 8  # Example value

    # Calling bit_stream_loader with k_bits and m_bits
    bit_stream = bit_stream_loader(filename, k_bits, m_bits)
    print('bit-stream shape: {}'.format(len(bit_stream)))
