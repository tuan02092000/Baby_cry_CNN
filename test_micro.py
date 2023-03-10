import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import write

import torch 
import torchaudio
import os

import config
import my_utils

# define sample rate
sample_rate = config.TARGET_SAMPLE_RATE

# define duration
duration = 5

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# target label
label = torch.tensor(0)  # baby cry

# network
net = torch.load("report/test_weights/Own_Aug_Mel_DB_Resnet18_22_2_no.pt")

# Mel transform
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.TARGET_SAMPLE_RATE, 
                                                                n_fft=config.N_FFT, 
                                                                hop_length=config.HOP_LENGTH, 
                                                                win_length=config.WIN_LENGTH, 
                                                                f_max=config.F_MAX,
                                                                n_mels=config.N_MELS,
                                                                normalized=config.NORMALIZE)
# MFCC transform
# mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=config.TARGET_SAMPLE_RATE)

# Loop
while True:  
    # recording
    myrecording = sd.rec(duration * sample_rate, samplerate=config.TARGET_SAMPLE_RATE, channels=1)
    print("[INFO] Start recording......")
    sd.wait()
    print("[INFO] End record.....")

    # save audio
    write('recording/recording_test.wav', sample_rate, myrecording)

    # my_recording = np.array(myrecording).reshape(1, -1)
    # signal = torch.tensor(my_recording)

    # read audio
    signal, sample_rate = torchaudio.load("recording/recording_test.wav")
    # print(signal)

    # resample audio
    if sample_rate != config.TARGET_SAMPLE_RATE:
        signal = torchaudio.transforms.Resample(sample_rate, config.TARGET_SAMPLE_RATE)(signal)
    
    # transform
    signal = mel_spectrogram(signal)
    # signal = mfcc_spectrogram(signal)

    # DB transform
    signal = torchaudio.transforms.AmplitudeToDB()(signal)
    print(signal)

    # convert input to cuda 
    input, target = signal.to(device), label.to(device)

    # expand dimension input
    input.unsqueeze_(0)

    # process
    predicted, expected = my_utils.predict(net, input, target, config.LABEL_DICT_CONVERT)
    print("[INFO] Predicted: {}".format(predicted))

    