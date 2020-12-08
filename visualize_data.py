import librosa
import numpy as np
import pywt
from scipy.signal import butter, lfilter
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fs = 44100
org_file_list = os.listdir('./small_database/')
scales = [2**i for i in range(1,11)] #dyadic scale
file_list = []
annotation = []

for index in range(len(org_file_list)):
  if org_file_list[index].endswith('.wav'):
    file_list.append(org_file_list[index])
  else:
    annotation.append(org_file_list[index])

annotation = sorted(annotation)
file_list = sorted(file_list)
###############################Band pass filter ################################
def butter_bandpass(lowcut,highcut,fs,order=9):
  nyq = 0.5*fs
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order,[low,high],btype='band')
  return b, a

def butter_bandpass_filter(data,lowcut,highcut,fs,order = 9):
  b, a = butter_bandpass(lowcut,highcut,fs,order)
  y = lfilter(b,a,data)
  return y
################################################################################
lowcut = 50 #Hz
highcut = 20000 #Hz
################################################################################
for index in range(len(file_list)):
  org_data, sample_rate = librosa.load(os.path.join('./small_database',file_list[index]),sr=fs)
  with open(os.path.join('./small_database',annotation[index]),'r') as f:
    file_labels = f.readlines()

  org_data = butter_bandpass_filter(org_data,lowcut,highcut,fs,order=6)

  for index in range(len(file_labels)):
    line = file_labels[index].strip('\n').split()
    time_st = float(line[0])
    time_ed = float(line[1])
    is_C = int(line[2])
    is_W = int(line[3])
    id_st = int(time_st*fs)
    id_ed = int(time_ed*fs)
    data = org_data[id_st:id_ed]
    org_cyc_length = id_ed - id_st + 1
    new_data = data
    new_data = new_data + np.finfo(np.float32).eps
    coefficient, frequencies = pywt.cwt(new_data,scales,'cmor1.5-1.0')
    coefficient = np.abs(coefficient)**2
    f_min = np.min(frequencies)
    f_max = np.max(frequencies)
    #t = np.arange(len(org_cyc_length))
    fig, ax = plt.subplots(nrows=4,figsize=(10,12),sharey=True)

    if is_C == 0 and is_W == 0:
      ax[0].set_title("Normal")
      img = ax[0].imshow(coefficient,extent=[0,org_cyc_length,f_min, f_max],aspect='auto',interpolation='nearest',
                   cmap='jet', norm=LogNorm(coefficient.min(),coefficient.max()))
      ax[0].spines['right'].set_visible(False)
      ax[0].spines['top'].set_visible(False)
      ax[0].set_ylabel('Scale')
      ax[0].set_xlabel('Time')
      fig.colorbar(img, ax=ax[0])
    elif is_C == 0 and is_W ==1:
      ax[1].set_title("Wheeze")
      ax[1].imshow(coefficient,extent=[0,org_cyc_length,f_min, f_max],aspect='auto',interpolation='nearest',
                   cmap='jet', norm=LogNorm(coefficient.min(),coefficient.max()))
      ax[1].spines['right'].set_visible(False)
      ax[1].spines['top'].set_visible(False)
      ax[1].set_ylabel('Scale')
      ax[1].set_xlabel('Time')
    elif is_C == 1 and is_W == 0:
      ax[2].set_title("Crackle")
      ax[2].imshow(coefficient,extent=[0,org_cyc_length,f_min, f_max],aspect='auto',interpolation='nearest',
                   cmap='jet', norm=LogNorm(coefficient.min(),coefficient.max()))
      ax[2].spines['right'].set_visible(False)
      ax[2].spines['top'].set_visible(False)
      ax[2].set_ylabel('Scale')
      ax[2].set_xlabel('Time')
    elif is_C == 1 and is_W == 1:
      ax[3].set_title("Both")
      ax[3].imshow(coefficient,extent=[0,org_cyc_length,f_min, f_max],aspect='auto',interpolation='nearest',
                   cmap='jet', norm=LogNorm(coefficient.min(),coefficient.max()))
      ax[3].spines['right'].set_visible(False)
      ax[3].spines['top'].set_visible(False)
      ax[3].set_ylabel('Scale')
      ax[3].set_xlabel('Time')

  #plt.tight_layout()

plt.show()

