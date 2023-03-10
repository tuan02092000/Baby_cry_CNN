import torch
import torchaudio

import seaborn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import time

import model
import dataset
import config
import my_utils

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dict = my_utils.make_data_path_list(config.AUDIO_DIR)
    train_data, val_data, test_data = my_utils.split_dataset(data_dict)

    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.TARGET_SAMPLE_RATE, 
    #                                                         n_fft=config.N_FFT, 
    #                                                         hop_length=config.HOP_LENGTH, 
    #                                                         win_length=config.WIN_LENGTH, 
    #                                                         f_max=config.F_MAX,
    #                                                         normalized=config.NORMALIZE,
    #                                                         n_mels=config.N_MELS)

    # train_dataset = dataset.AudioDataset(train_data, transforms=mel_spectrogram, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="train")
    # val_dataset = dataset.AudioDataset(val_data, transforms=mel_spectrogram, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="val")
    # test_dataset = dataset.AudioDataset(test_data, transforms=mel_spectrogram, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="test")
    
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
                                                            normalized=config.NORMALIZE,
                                                            n_mels=config.N_MELS,
    )

    # mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=config.TARGET_SAMPLE_RATE)

    train_dataset = dataset.AudioDataset(train_data, transforms=mel_spectrogram, augumentations=augumentations,target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="train")
    val_dataset = dataset.AudioDataset(val_data, transforms=mel_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="val")
    test_dataset = dataset.AudioDataset(test_data, transforms=mel_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="test")

    num_audio = len(test_dataset)
    print("Num audio:", num_audio)
    
    start = time.time()

    net = torch.load(config.PATH_TO_MODEL)

    # index_test = 50
    # print("[INFO] Path to audio {} is {}".format(index_test, test_data["path"][index_test]))
    # print(test_dataset[index_test][0].to(device))
    # print(test_dataset[index_test][1].to(device))
    # input, target = test_dataset[index_test][0].to(device), test_dataset[index_test][1].to(device)
    # input.unsqueeze_(0)

    # predicted, expected = my_utils.predict(net, input, target, config.LABEL_DICT_CONVERT)
    # print("Predicted: {}, Expected: {}".format(predicted, expected))

    y_true = []
    y_pred = []

    num_correct = 0
    for index_test in range(num_audio):
        input, target = test_dataset[index_test][0].to(device), test_dataset[index_test][1].to(device)
        input.unsqueeze_(0)

        predicted, expected = my_utils.predict(net, input, target, config.LABEL_DICT_CONVERT)

        # if predicted == expected:
        #     num_correct += 1

        y_true.append(config.LABEL_DICT[expected])
        y_pred.append(config.LABEL_DICT[predicted])

    end = time.time()

    print("[INFO] Average time: ", (end - start) / num_audio)
        
    
    # print("Num correct: {}, total: {}".format(num_correct, len(test_dataset)))

    # print("Label: ", y_true)
    # print("Pred: ", y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)

    y_true_convert = []
    y_pred_convert = []

    # Convert results for evaluation
    for i, value in enumerate(y_true):
        if value == 0:
            y_true_convert.append(1)
        else:
            y_true_convert.append(0)

    for i, value in enumerate(y_pred):
        if value == 0:
            y_pred_convert.append(1)
        else:
            y_pred_convert.append(0)

    print("y_true convert: ", y_true_convert)
    print("y_pred convert: ", y_pred_convert)

    # Accuracy
    accuracy = accuracy_score(y_true_convert, y_pred_convert)
    print("[INFO] Accuracy: %.4f" % accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_true_convert, y_pred_convert)
    print(cm)
    sea_cm = seaborn.heatmap(cm, annot=True, xticklabels=config.LABEL_DICT.values(), yticklabels=config.LABEL_DICT.values(), cmap='YlGnBu')
    sea_cm.figure.savefig("confusion_matrix.png")

    # TP / FP / TN / FN
    TN, FP, FN, TP = confusion_matrix(y_true_convert, y_pred_convert).ravel()
    print('[INFO] True Positive: %d' % TP)
    print('[INFO] False Positive: %d' % FP)
    print('[INFO] False Negative: %d' % FN)
    print('[INFO] True Negative: %d' % TN)

    # Precision
    precision = precision_score(y_true_convert, y_pred_convert)
    print("[INFO] Precision Score: %.4f" % precision)

    # Recall
    recall = recall_score(y_true_convert, y_pred_convert)
    print("[INFO] Recall: %.4f" % recall)

    # F1 Score
    f1 = f1_score(y_true_convert, y_pred_convert)
    print("[INFO] F1 score: %.4f" % f1)




