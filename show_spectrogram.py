import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
import cv2

# Path to audio
# audio_file = "report/test_audio/test_baby_cry_2.wav"
# audio_file = "ESC-50-master/audio/1-104089-A-22.wav" # clapping
audio_file = "own_dataset/baby_cry/c6fd9d60-0fa7-44c0-b3ce-0192527d7b81-1430038968282-1.7-m-04-hu.wav" 

# Load audio
samples, sample_rate = librosa.load(audio_file)
# print(samples, sample_rate)
sample_rate = 32000
# Waveform
plt.figure(figsize=(15, 4))
display.waveshow(samples, sr=sample_rate)
plt.show()

# Spectrogram
plt.figure(figsize=(15, 4))
sgram = librosa.stft(samples)
display.specshow(sgram)
plt.show()

# Mel spectrogram
plt.figure(figsize=(15, 4))
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
# print(mel_scale_sgram.shape)
# mel_scale_sgram = np.array(mel_scale_sgram)

# mel_scale_sgram = mel_scale_sgram * 255
# mel_scale_sgram = np.log2(mel_scale_sgram)
# cv2.imwrite("mel_cv2.jpg", mel_scale_sgram * 255)
# image = Image.fromarray(mel_scale_sgram, 'L')

# print(image)
# image.save("clapping_image.jpg")
# image.show()
# plt.pause(10)
librosa.display.specshow(mel_scale_sgram, cmap='gray')
# librosa.display.specshow(mel_scale_sgram, x_axis='time',
#                         #    y_axis='log', hop_length=256, sr=sample_rate,
#                             y_axis='hz', hop_length=256, sr=sample_rate,
#                              cmap='gray')
# img = np.array(img)

# image_gray = cv2.imread("image_mel_2.jpg", 0)
# cv2.imwrite("image_mel_2.jpg", image_gray)
# plt.imsave("image_mel_npdb_2.jpg", image)
plt.show()

# Mel spectrogram dB
plt.figure()
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
# mel_sgram = np.array(mel_sgram)

# image = Image.fromarray(mel_sgram , 'L')
# image.show()
librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()

# MFCC

# MFCC original
mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
plt.figure(figsize=(15, 4))
librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
plt.show()

# MFCC dB
plt.figure(figsize=(15, 4))
mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.min)
librosa.display.specshow(mfcc_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.show()

##################

# mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
# mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.min)
# mel_scale_sgram = np.log2(mel_scale_sgram)
# cv2.imwrite("mfcc_db_cv2.jpg", mfcc_db)

# sgram_mag, _ = librosa.magphase(sgram)
# magnitude, phase = librosa.magphase(sgram)
# print(magnitude, phase)
# mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_mels=256, hop_length=256)
# mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
# mel_sgram = np.array(mel_sgram)
# cv2.imwrite("MEL_DB_32k_hop.jpg", mel_sgram)
