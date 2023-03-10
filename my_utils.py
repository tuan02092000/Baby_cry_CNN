import os
import torch
import torchaudio
from sklearn.model_selection import train_test_split
import config
import matplotlib.pyplot as plt

# def make_data_path_list(audio_dir):
#     data_dict = {"path": [], "label": []}
#     for label in os.listdir(audio_dir):
#         path_to_label = os.path.join(audio_dir, label)
#         for audio in os.listdir(path_to_label):
#             data_dict["path"].append(os.path.join(path_to_label, audio))
#             data_dict["label"].append(label)
#     return data_dict

# 50 classes of ESC50
# def make_data_path_list(path_audio):
# 	data_dict = {"path": [], "label": []}
# 	for index, audio in enumerate(os.listdir(path_audio)):
# 		fold, clip_id, take, target = audio.split(".")[0].split("-")
# 		data_dict["path"].append(os.path.join(path_audio, audio))
# 		data_dict["label"].append(int(target))
# 	return data_dict

# 2 classes of ESC50: baby and other
# def make_data_path_list(path_audio):
#     data_dict = {"path": [], "label": []}
#     for index, audio in enumerate(os.listdir(path_audio)):
#         fold, clip_id, take, target = audio.split(".")[0].split("-")
#         data_dict["path"].append(os.path.join(path_audio, audio))
#         if int(target) == 20:
#             data_dict["label"].append(0)
#         else:
#             data_dict["label"].append(1)
#     return data_dict

# 2 classes from folder
def make_data_path_list(path_audio):
    data_dict = {"path": [], "label": []}
    for label in os.listdir(path_audio):
        path_to_folder = os.path.join(path_audio, label)
        for audio in os.listdir(path_to_folder):
            data_dict['path'].append(os.path.join(path_to_folder, audio))
            data_dict['label'].append(config.LABEL_DICT[label])        
    return data_dict

def split_dataset(data_dict):
    train_audio, val_test_audio, train_label, val_test_label = train_test_split(data_dict["path"], data_dict["label"], test_size=0.2, random_state=42, shuffle=True)
    val_audio, test_audio, val_label, test_label = train_test_split(val_test_audio, val_test_label, test_size=0.2, random_state=42, shuffle=True)
    train_data = {"path": train_audio, "label": train_label}
    val_data = {"path": val_audio, "label": val_label}
    test_data = {"path": test_audio, "label": test_label}
    return train_data, val_data, test_data

def save_model(model, name_model):
    torch.save(model, os.path.join(config.PATH_TO_SAVE_MODEL, name_model))

def predict(net, input, target, label_dict_convert):
    net.eval()
    with torch.no_grad():
        prediction = net(input)
        predict_index = prediction[0].argmax(0)
        predicted = label_dict_convert[predict_index.cpu().item()]
        expected = label_dict_convert[target.cpu().item()]
    return predicted, expected

def plot_history(history):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.PATH_TO_FIGURE)

def plot_waveform(waveform, sample_rate, label):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(label)
    plt.show(block=False)
    plt.pause(5)

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)
    plt.pause(5)

def show_spectrogram(waveform_classA, waveform_classB):
    yes_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classA)
    print("\nShape of yes spectrogram: {}".format(yes_spectrogram.size()))
    
    no_spectrogram = torchaudio.transforms.Spectrogram()(waveform_classB)
    print("Shape of no spectrogram: {}".format(no_spectrogram.size()))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Features of {}".format('no'))
    plt.imshow(yes_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.subplot(1, 2, 2)
    plt.title("Features of {}".format('yes'))
    plt.imshow(no_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')  

def show_melspectrogram(waveform,sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mel_spectrogram.size()))

    plt.figure()
    plt.imshow(mel_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')

def show_mfcc(waveform,sample_rate):
    mfcc_spectrogram = torchaudio.transforms.MFCC(sample_rate= sample_rate)(waveform)
    print("Shape of spectrogram: {}".format(mfcc_spectrogram.size()))

    plt.figure()
    fig1 = plt.gcf()
    plt.imshow(mfcc_spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    
    plt.figure()
    plt.plot(mfcc_spectrogram.log2()[0,:,:].numpy())
    plt.draw()

if __name__ == "__main__":
    data_dict = make_data_path_list(config.AUDIO_DIR)
    print("[INFO] Len total data: {}".format(len(data_dict["path"])))
    train_data, val_data, test_data = split_dataset(data_dict)
    print(train_data)
    # print(train_data["path"])
    print("[INFO] Len train data: {}".format(len(train_data["path"])))
    print("[INFO] Len val data: {}".format(len(val_data["path"])))
    print("[INFO] Len test data: {}".format(len(test_data["path"])))