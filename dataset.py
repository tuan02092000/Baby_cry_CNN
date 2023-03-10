import torch
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
import my_utils
import config
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)

class AudioDataset(Dataset):
    def __init__(self, dataset, transforms, augumentations, target_sample_rate, num_samples, device, phase):
        self.dataset = dataset
        self.transforms = transforms
        self.augmentations = Compose(augumentations)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
        self.phase = phase   
    def __len__(self):
        return len(self.dataset["path"])
    def __getitem__(self,index):
        audio_sample_path = self.dataset["path"][index]
        label = self.dataset["label"][index]
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = self._resample(signal, sample_rate)
        # signal = self._mix_down(signal)

        if self.phase != 'test':
            signal = self._cut(signal)
        # signal = self._cut(signal)

        # signal = self._right_pad(signal)
        if self.augmentations != None and self.phase == 'train':
            signal = self.augmentations(signal)
        if self.transforms != None:
            signal = self.transforms(signal)
            signal = torchaudio.transforms.AmplitudeToDB()(signal)
        # label = torch.tensor(config.CLASS_MAPPING.index(label))
        label = torch.tensor(label)
        return signal, label
    def _resample(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepDim=True)
        return signal
    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    def _right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_sample = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_sample)
            signal = torch.nn.function.pad(signal, last_dim_padding)
        return signal

if __name__ == "__main__":
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device: {}".format(device))

    data_dict = my_utils.make_data_path_list(config.AUDIO_DIR)
    train_data, val_data, test_data = my_utils.split_dataset(data_dict)

    augumentations = [
            RandomResizedCrop(n_samples=config.NUM_SAMPLES),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=config.TARGET_SAMPLE_RATE)], p=0.8),
            RandomApply([Delay(sample_rate=config.TARGET_SAMPLE_RATE)], p=0.5),
            RandomApply([PitchShift(n_samples=config.NUM_SAMPLES, sample_rate=config.TARGET_SAMPLE_RATE)], p=0.4),
            RandomApply([Reverb(sample_rate=config.TARGET_SAMPLE_RATE)], p=0.3),
        ]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.TARGET_SAMPLE_RATE, 
                                                            n_fft=config.N_FFT, 
                                                            hop_length=config.HOP_LENGTH, 
                                                            win_length=config.WIN_LENGTH, 
                                                            f_max=config.F_MAX,
                                                            n_mels=config.N_MELS,
                                                            normalized=config.NORMALIZE)

    train_dataset = AudioDataset(train_data, transforms=mel_spectrogram, augumentations=augumentations,target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="train")
    val_dataset = AudioDataset(val_data, transforms=mel_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="val")
    test_dataset = AudioDataset(test_data, transforms=mel_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="test")



    index_test = 10
    print("[INFO] Train index {} is {}, {}".format(index_test, train_dataset[index_test][0], train_dataset[index_test][1]))
    print("[INFO] Val index {} is {}, {}".format(index_test, val_dataset[index_test][0], val_dataset[index_test][1]))
    print("[INFO] Test index {} is {}, {}".format(index_test, test_dataset[index_test][0], test_dataset[index_test][1]))

