# Melspectrogram DB image

import cv2
import os
import librosa
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

# config
# num_samples = 100000
target_sample_rate = 44100
window_length = 2205
hop_length = 308
n_mels = 224
f_max = 18000
n_fft = 4096
normalize = True

num_samples = 5 * target_sample_rate # 5*44100

def prepare_folder(root_dataset_image, labels):
    if not os.path.exists(root_dataset_image):
        os.makedirs(root_dataset_image)
    for label in labels:
        if not os.path.exists(os.path.join(root_dataset_image, label)):
            os.mkdir(os.path.join(root_dataset_image, label))

def prepare_dataset_image(root_dataset_audio, root_dataset_image):
    for label_folder in os.listdir(root_dataset_audio):
        path_to_folder_audio = os.path.join(root_dataset_audio, label_folder)
        path_to_folder_image = os.path.join(root_dataset_image, label_folder)
        for audio in os.listdir(path_to_folder_audio):
            path_to_audio = os.path.join(path_to_folder_audio, audio)

            # one image
            # img = generate_melspectrogram(path_to_audio)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # path_to_save_image = os.path.join(path_to_folder_image, audio.replace('.wav', '.jpg'))
            # cv2.imwrite(path_to_save_image, img)

            # list image
            list_img = generate_list_melspectrogram(path_to_audio)
            for index, img in enumerate(list_img):
                name_audio = audio[:-4] + "_{}.jpg".format(index)
                path_save_image = os.path.join(path_to_folder_image, name_audio)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.array(img)
                print(img.shape)
                cv2.imwrite(path_save_image, img)
    print("[INFO] Prepare dataset done")

def generate_melspectrogram(audio_file):
    samples, sample_rate = librosa.load(audio_file)
    print(samples.shape)   
    if sample_rate != target_sample_rate:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sample_rate)
    if samples.shape[0] > num_samples:
        samples = samples[:num_samples]
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=target_sample_rate, n_mels=n_mels)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    mel_sgram = np.array(mel_sgram)
    return mel_sgram

def generate_list_melspectrogram(audio_file):
    list_mel_sgram = []
    step = 0
    samples, sample_rate = librosa.load(audio_file)
    if sample_rate != target_sample_rate:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sample_rate)
    for i in range(len(samples) // num_samples * 2 - 1):
        s_i = samples[step:step + num_samples]
        step += num_samples // 2
        sgram = librosa.stft(s_i)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=target_sample_rate, n_mels=n_mels)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        list_mel_sgram.append(mel_sgram)
    s_last = samples[step:]
    s_last = np.pad(s_last, (0, num_samples - len(s_last)), mode='constant')
    sgram = librosa.stft(s_last)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=target_sample_rate, n_mels=n_mels)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    list_mel_sgram.append(mel_sgram)
    return list_mel_sgram

def generate_melspectrogram_torchaudio(audio_file):
    signal, sample_rate = torchaudio.load(audio_file)

    # Resample
    if sample_rate != target_sample_rate:
        signal = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(signal)

    # Cut
    if signal.shape[1] > num_samples * 5:
        signal = signal[:, :num_samples]
    
    # Right pad
    if signal.shape[1] < num_samples:
        num_missing_sample = num_samples- signal.shape[1]
        last_dim_padding = (0, num_missing_sample)
        signal = torch.nn.function.pad(signal, last_dim_padding)

    # transform 
    # signal = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, 
    #                                                         n_fft=n_fft, 
    #                                                         hop_length=hop_length, 
    #                                                         win_length=window_length, 
    #                                                         f_max=f_max,
    #                                                         n_mels=n_mels,
    #                                                         normalized=normalize)(signal)
    # mel_spectrogram = signal.log2()[0,:,:].numpy()
    # plt.figure()
    # plt.imshow(mel_spectrogram, cmap='viridis')
    # plt.pause(20)

    signal = torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, 
                                                            n_fft=n_fft, 
                                                            hop_length=hop_length, 
                                                            win_length=window_length, 
                                                            f_max=f_max,
                                                            n_mels=n_mels,
                                                            normalized=normalize)(signal)
    signal = torchaudio.transforms.AmplitudeToDB()(signal)
    signal = np.abs(signal[0])
    
    return np.array(signal)

if __name__ == '__main__':
    # dataset audio
    root_dataset_audio = 'own_dataset'
    # dataset image
    root_dataset_image = 'own_dataset_image'
    # labels datasets
    labels = ['baby_cry', 'other_sound']   

    # create folder
    prepare_folder(root_dataset_image, labels)

    # generate dataset
    prepare_dataset_image(root_dataset_audio, root_dataset_image)

    







