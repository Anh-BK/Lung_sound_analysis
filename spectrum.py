import numpy as np
import matplotlib.pyplot as plt 
import pywt
from scipy.io import wavfile as wav
from scipy import signal
from scipy.fftpack import fft, fftshift
import soundfile as sf
import librosa

data, rate = librosa.load('101_1b1_Al_sc_Meditron.wav')
print(data.shape)
#fft
#N_FFT = 2048
#fft_out = fft(data,N_FFT)
#freq = np.linspace(-rate/2,rate/2,len(fft_out))
#visualize
#plt.plot(freq, np.abs(fftshift(fft_out)))
#plt.show()
