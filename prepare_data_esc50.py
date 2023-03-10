import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import write

import config
import shutil

def make_data_path_list(path_audio):
    for index, audio in enumerate(os.listdir(path_audio)):
        fold, clip_id, take, target = audio.split(".")[0].split("-")
        path_to_audio = os.path.join(path_audio, audio)
        if int(target) == 20:
            shutil.copy(path_to_audio, "/home/nguyen-tuan/VNPT/CAM/sound_cam/own_dataset/baby_cry")
        else:
            shutil.copy(path_to_audio, "/home/nguyen-tuan/VNPT/CAM/sound_cam/own_dataset/other_sound")

if __name__ == '__main__':
    make_data_path_list(config.AUDIO_DIR)