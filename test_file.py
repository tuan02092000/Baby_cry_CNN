import torch
import torchaudio
import os

import model
import dataset
import config
import my_utils

import sounddevice as sd
from scipy.io.wavfile import write


if __name__ == '__main__':
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # audio path
    audio_sample_path = "report/test_audio/alarm_sound.wav"

    # read audio
    signal, sample_rate = torchaudio.load(audio_sample_path)
    # print(len(signal[0]))

    # resample audio
    if sample_rate != config.TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, config.TARGET_SAMPLE_RATE)
        signal = resampler(signal)

    # Mel transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.TARGET_SAMPLE_RATE, 
                                                            n_fft=config.N_FFT, 
                                                            hop_length=config.HOP_LENGTH, 
                                                            win_length=config.WIN_LENGTH, 
                                                            f_max=config.F_MAX,
                                                            normalized=config.NORMALIZE,
                                                            n_mels=config.N_MELS)
    signal = mel_spectrogram(signal)

    # MFCC transform
    # mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=config.TARGET_SAMPLE_RATE, )
    # signal = mfcc_spectrogram(signal)

    # DB transform
    signal = torchaudio.transforms.AmplitudeToDB()(signal)
    print(signal)

    # Label target
    label = torch.tensor(0)  # baby cry

    # label = torch.tensor(0)  # dog

    # label = torch.tensor(5)  # cat

    # label = torch.tensor(26)  # laughing

    # label = torch.tensor(37)  # clock alarm

    # load model
    net = torch.load(config.PATH_TO_MODEL)

    # put input to device
    input, target = signal.to(device), label.to(device)

    # expand dimension input
    input.unsqueeze_(0)

    # Process
    predicted, expected = my_utils.predict(net, input, target, config.LABEL_DICT_CONVERT)
    # print("Predicted: {}, Expected: {}".format(predicted, expected))

print("[INFO] Audio:", audio_sample_path)
print("[INFO] Predicted: {}".format(predicted))