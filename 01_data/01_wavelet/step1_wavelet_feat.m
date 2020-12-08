
% 01/ Draw data directory %TODO
opts.data_txt   =  './../../01_data_trip_loss/data/txt_file/'
opts.data_audio =  './../../01_data_trip_loss/data/audio_file/'

% 02/ Saved whole spectrogram directory %TODO
mkdir('./../11_wavelet_10s')
mkdir('./../11_wavelet_10s/01_W')
mkdir('./../11_wavelet_10s/02_C')
mkdir('./../11_wavelet_10s/03_B')
mkdir('./../11_wavelet_10s/04_N')

opts.store_dir_W  =  './../11_wavelet_10s/01_W/';
opts.store_dir_C  =  './../11_wavelet_10s/02_C/';
opts.store_dir_B  =  './../11_wavelet_10s/03_B/';
opts.store_dir_N  =  './../11_wavelet_10s/04_N/';

% 03/ Call function to generate spectrogram
fprintf('================================== Creating Spectrogram \n');
get_data_wavelet(opts);
