### Intro:
To run the code that excutes the TX and RX pipelines for Alice and Bob:
1. Clone this repository "git clone ..."
2- install tensorflow 2.x
3- run main.py from the terminal using this command "python3 main.py <Modulation> <Q-bits> <SNR>"

The code accepts 3 parameters:
* Modulation: Type of modulation to use for each carrier. Possible values [QPSK, QAM_16, QAM_64]
* Q-bit: Number of Quantization bits. Possible values [4, 8, 12]
* SNR: Signal-to-Noise ratio. Possible values [10, 12, 14 , ..., 30] 

	



##### Complete run command example:
```
> python3 main.py QPSK 8 20
```
