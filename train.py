import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import copy

import dataset
import dataloader
import model
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

def train_model(model, criterion, optimier, scheduler, dataloader_dict, train_dataset, val_dataset, device):
    print('[INFO] Start training network...')

    start = time.time()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(config.EPOCHS):
        print('\n[INFO] Epoch {}/{}'.format(epoch, config.EPOCHS - 1))
        print('-' * 50)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels in tqdm(dataloader_dict[phase]):
                images = images.to(device)
                # print(images.shape)
                labels = labels.to(device)
                optimier.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predicts = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimier.step()
                        # scheduler.step()
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(predicts == labels.data)

            if phase == 'train':
                scheduler.step()

                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)
                H["train_loss"].append(epoch_loss)
                H["train_acc"].append(epoch_acc.cpu().detach().numpy())
            else:
                epoch_loss = running_loss / len(val_dataset)
                epoch_acc = running_corrects.double() / len(val_dataset)
                H["val_loss"].append(epoch_loss)
                H["val_acc"].append(epoch_acc.cpu().detach().numpy())

            print('\n[INFO] {} Loss: {:4f}, Acc: {:4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())      
             
    time_elapsed = time.time() - start
    print(f'\n[INFO] Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\n[INFO] Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    my_utils.save_model(model, config.NAME_SAVE_MODEL)
    my_utils.plot_history(H) 
   

if __name__ == '__main__':
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device: {}".format(device))
    
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

    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.TARGET_SAMPLE_RATE, 
    #                                                         n_fft=config.N_FFT, 
    #                                                         hop_length=config.HOP_LENGTH, 
    #                                                         win_length=config.WIN_LENGTH, 
    #                                                         f_max=config.F_MAX,
    #                                                         normalized=config.NORMALIZE,
    #                                                         n_mels=config.N_MELS)

    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate=config.TARGET_SAMPLE_RATE)

    train_dataset = dataset.AudioDataset(train_data, transforms=mfcc_spectrogram, augumentations=augumentations,target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="train")
    val_dataset = dataset.AudioDataset(val_data, transforms=mfcc_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="val")
    test_dataset = dataset.AudioDataset(test_data, transforms=mfcc_spectrogram, augumentations=augumentations, target_sample_rate=config.TARGET_SAMPLE_RATE, num_samples=config.NUM_SAMPLES, device=device, phase="test")


    dataloader_dict = dataloader.get_dataloader_dict(train_dataset, val_dataset, test_dataset, batch_size=config.BATCH_SIZE)

    net = model.Resnet18()
    # net = model.AudioClassifier()

    # transfer learning
    # net = torch.load(config.PATH_TO_MODEL)

    criterion = nn.CrossEntropyLoss()
    # criterion = F.nll_loss()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=config.LR)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dataset)),
                                                epochs=config.EPOCHS,
                                                anneal_strategy='linear')

    train_model(net, criterion, optimizer, scheduler, dataloader_dict, train_dataset, val_dataset, device)