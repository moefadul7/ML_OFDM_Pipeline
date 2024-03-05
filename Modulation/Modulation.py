import textwrap
import numpy as np


class OFDM:
    def __init__(self,K,P,CP,pilotValue):
        self.K = K
        self.CP = CP
        self.pilotValue = pilotValue
        self.total_Set = np.arange(K)
        pilots = self.total_Set[::K//P]
        self.pilots = np.append(pilots, self.total_Set[-1])
        self.data_carriers = np.delete(self.total_Set, self.pilots)
        self.P = P + 1


class QAM_16:
    def __init__(self, D):
        self.mu = 4
        self.D = D
        self.bits_per_symbol = D * self.mu
        self.mapper =  {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -3-1j,
        (0,0,1,0) : -3+3j,
        (0,0,1,1) : -3+1j,
        (0,1,0,0) : -1-3j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : -1+3j,
        (0,1,1,1) : -1+1j,
        (1,0,0,0) :  3-3j,
        (1,0,0,1) :  3-1j,
        (1,0,1,0) :  3+3j,
        (1,0,1,1) :  3+1j,
        (1,1,0,0) :  1-3j,
        (1,1,0,1) :  1-1j,
        (1,1,1,0) :  1+3j,
        (1,1,1,1) :  1+1j
        }
        self.demapper = {i : j for j, i in self.mapper.items()}

class QAM_64:
    def __init__(self, D):
        self.mu = 6
        self.D = D
        self.bits_per_symbol = D * self.mu
        self.mapper = {
        (0,0,0,0,0,0) : -7-7j,
        (0,0,0,0,0,1) : -7-5j,
        (0,0,0,0,1,0) : -7-1j,
        (0,0,0,0,1,1) : -7-3j,
        (0,0,0,1,0,0) : -7+7j,
        (0,0,0,1,0,1) : -7+5j,
        (0,0,0,1,1,0) : -7+1j,
        (0,0,0,1,1,1) : -7+3j,
        (0,0,1,0,0,0) :  -5-7j,
        (0,0,1,0,0,1) :  -5-5j,
        (0,0,1,0,1,0) :  -5-1j,
        (0,0,1,0,1,1) :  -5-3j,
        (0,0,1,1,0,0) :  -5+7j,
        (0,0,1,1,0,1) :  -5+5j,
        (0,0,1,1,1,0) :  -5+1j,
        (0,0,1,1,1,1) :  -5+3j,
        (0, 1, 0, 0, 0, 0): -1 - 7j,
        (0, 1, 0, 0, 0, 1): -1 - 5j,
        (0, 1, 0, 0, 1, 0): -1 - 1j,
        (0, 1, 0, 0, 1, 1): -1 - 3j,
        (0, 1, 0, 1, 0, 0): -1 + 7j,
        (0, 1, 0, 1, 0, 1): -1 + 5j,
        (0, 1, 0, 1, 1, 0): -1 + 1j,
        (0, 1, 0, 1, 1, 1): -1 + 3j,
        (0, 1, 1, 0, 0, 0): -3 - 7j,
        (0, 1, 1, 0, 0, 1): -3 - 5j,
        (0, 1, 1, 0, 1, 0): -3 - 1j,
        (0, 1, 1, 0, 1, 1): -3 - 3j,
        (0, 1, 1, 1, 0, 0): -3 + 7j,
        (0, 1, 1, 1, 0, 1): -3 + 5j,
        (0, 1, 1, 1, 1, 0): -3 + 1j,
        (0, 1, 1, 1, 1, 1): -3 + 3j,
        (1, 0, 0, 0, 0, 0): 7 - 7j,
        (1, 0, 0, 0, 0, 1): 7 - 5j,
        (1, 0, 0, 0, 1, 0): 7 - 1j,
        (1, 0, 0, 0, 1, 1): 7 - 3j,
        (1, 0, 0, 1, 0, 0): 7 + 7j,
        (1, 0, 0, 1, 0, 1): 7 + 5j,
        (1, 0, 0, 1, 1, 0): 7 + 1j,
        (1, 0, 0, 1, 1, 1): 7 + 3j,
        (1, 0, 1, 0, 0, 0): 5 - 7j,
        (1, 0, 1, 0, 0, 1): 5 - 5j,
        (1, 0, 1, 0, 1, 0): 5 - 1j,
        (1, 0, 1, 0, 1, 1): 5 - 3j,
        (1, 0, 1, 1, 0, 0): 5 + 7j,
        (1, 0, 1, 1, 0, 1): 5 + 5j,
        (1, 0, 1, 1, 1, 0): 5 + 1j,
        (1, 0, 1, 1, 1, 1): 5 + 3j,
        (1, 1, 0, 0, 0, 0): 1 - 7j,
        (1, 1, 0, 0, 0, 1): 1 - 5j,
        (1, 1, 0, 0, 1, 0): 1 - 1j,
        (1, 1, 0, 0, 1, 1): 1 - 3j,
        (1, 1, 0, 1, 0, 0): 1 + 7j,
        (1, 1, 0, 1, 0, 1): 1 + 5j,
        (1, 1, 0, 1, 1, 0): 1 + 1j,
        (1, 1, 0, 1, 1, 1): 1 + 3j,
        (1, 1, 1, 0, 0, 0): 3 - 7j,
        (1, 1, 1, 0, 0, 1): 3 - 5j,
        (1, 1, 1, 0, 1, 0): 3 - 1j,
        (1, 1, 1, 0, 1, 1): 3 - 3j,
        (1, 1, 1, 1, 0, 0): 3 + 7j,
        (1, 1, 1, 1, 0, 1): 3 + 5j,
        (1, 1, 1, 1, 1, 0): 3 + 1j,
        (1, 1, 1, 1, 1, 1): 3 + 3j
        }
        self.demapper = {i : j for j, i in self.mapper.items()}

class QPSK:
    def __init__(self, D):
        self.mu = 2
        self.D = D
        self.bits_per_symbol = D * self.mu
        self.mapper =  {
        (0,0) : -1-1j,
        (0,1) : -1+1j,
        (1,0) : 1-1j,
        (1,1) : 1+1j,
        }
        self.demapper = {i : j for j, i in self.mapper.items()}

def Serial_to_Parallel(bit_stream, Mapper):
    # partitioned_bits = bit_stream.split(" ")
    # partitioned_bits = np.array([ int(x) for x in partitioned_bits])
    partitioned_bits = bit_stream.copy()
    # zero pad depending on bit_per_symbol & the total number of bits in the bit stream
    Remainder = Mapper.bits_per_symbol - (len(partitioned_bits) % Mapper.bits_per_symbol)
    partitioned_bits.resize(len(partitioned_bits) + Remainder, refcheck=False)

    # reshape bit_stream to symbols_per_stream X bits_per_symbol X bits_per_carrier
    partitioned_bits = partitioned_bits.reshape(len(partitioned_bits)//Mapper.bits_per_symbol, Mapper.bits_per_symbol)
    bits_SP = []
    for symbol in partitioned_bits:
        bits_SP.append(symbol.reshape(Mapper.D,Mapper.mu))

    bits_SP = np.array(bits_SP)
    return bits_SP.shape[0],bits_SP


def Bits_to_Symbols(bits_SP, Mapper):
    Mapped_Symbols = []
    for symbol in bits_SP:
        mapped_symbol = np.array([Mapper.mapper[tuple(word)] for word in symbol])
        Mapped_Symbols.append(mapped_symbol)
    return np.array(Mapped_Symbols)



def OFDM_symbol(payloads, ofdm):
    Symbols = []
    for payload in payloads:
        symbol = np.zeros(ofdm.K, dtype=complex)  # the overall K subcarriers
        symbol[ofdm.pilots] = ofdm.pilotValue   # allocate the pilot subcarriers
        symbol[ofdm.data_carriers] = payload   # allocate the pilot subcarriers
        Symbols.append(symbol)
    return np.array(Symbols)

def OFDM_time(payloads):
    Symbols_IFFT = []
    for payload in payloads:
        symbol_fft = np.fft.ifft(payload)
        Symbols_IFFT.append(symbol_fft)
    return np.array(Symbols_IFFT)


def WithCP(OFDM_symbols, ofdm):
    Symbols_CP = []
    for symbol in OFDM_symbols:
        cp = symbol[-ofdm.CP:]               # take the last CP samples ...
        symbol_CP = np.hstack([cp, symbol])  # ... and add them to the beginning
        Symbols_CP.append(symbol_CP)
    return np.array(Symbols_CP)


def WithoutCP(Received_signals, ofdm):
    WithoutCP_Signals = []
    for signal in Received_signals:
        WithoutCP_Signals.append(signal[ofdm.CP:(ofdm.CP+ofdm.K)])
    return np.array(WithoutCP_Signals)


def OFDM_freq(Received_No_CP):
    Received_FFT = []
    for payload in Received_No_CP:
        symbol_fft = np.fft.fft(payload)
        Received_FFT.append(symbol_fft)
    return np.array(Received_FFT)

def Extract_payloads(Equalized_OFDM, ofdm):
    return Equalized_OFDM[:,ofdm.data_carriers]


def Demodulation(Est_Symbols, Mapper):
    # All possible constellation points
    Const = np.array([x for x in Mapper.demapper.keys()])

    # Calculate the diff between each sample and all possible const points
    bits_per_symbol = []
    for symbol in Est_Symbols:
        dist = abs(symbol.reshape(-1,1) - Const.reshape(1,-1))
        dist_idx = dist.argmin(axis=1)
        Decisions = Const[dist_idx]
        symbol_to_bits = [Mapper.demapper[d] for d in Decisions]
        bits_per_symbol.append(symbol_to_bits)
    return np.array(bits_per_symbol)

def Parallel_to_Serial(Demod_symbols):
    return Demod_symbols.reshape(-1)

def BER(Stream1, Stream2):
    span = np.min([len(Stream1), len(Stream2)])
    print("\n Len of Tx bit_stream is {} and Rx bit_stream is {}.\n The span = {}".format(len(Stream1), len(Stream2), span))
    return np.sum(abs(Stream1[:span] - Stream2[:span]))/span, span
