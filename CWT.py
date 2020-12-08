import numpy as np
import pywt
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

data, rate = librosa.load('101_1b1_Al_sc_Meditron.wav') # read file
time = np.arange(0, len(data))/rate # time interval of the signal
dt = 1./rate 
scales = np.arange(44,221) 
f = pywt.scale2frequency('cmor1.5-1.0', scales)/dt
print('rate: {} (Herzt)'.format(rate))
print('Frequencies: {} (Herzt)'.format(f))
coefficient, frequencies = pywt.cwt(data, scales, 'cmor1.5-1.0')
power = (np.abs(coefficient))**2 #power of continuous wavelet transform
levels = [0.065, 0.125, 0.25, 0.5, 1, 2, 4, 8]
coutourlevels = np.log2(levels)
#visual data
title ='Wavelet Transform (Power Spectrum) of Signal' 
ylabel = 'Frequency'
xlabel = 'Time'

fig, ax = plt.subplots(figsize = (10,10))
im = ax.contourf(time, np.log2(frequencies), np.log2(power), coutourlevels, extend = 'both', cmap = plt.cm.seismic)
ax.set_title(title, fontsize = 20)
ax.set_ylabel(ylabel, fontsize =18)
ax.set_xlabel(xlabel, fontsize = 18)
yticks = 2**np.arange(np.ceil(np.log2(frequencies.min())),np.ceil(np.log2(frequencies.max()) + 1))
ax.set_yticks(np.log2(yticks))
ax.set_yticklabels(yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], -1)
plt.show()
