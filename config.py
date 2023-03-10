# AUDIO_DIR = "/home/nguyen-tuan/VNPT/CAM/sound_cam/VoxCeleb_gender/VoxCeleb_gender"
# AUDIO_DIR = "/home/nguyen-tuan/VNPT/CAM/sound_cam/ESC-50-master/audio"
AUDIO_DIR = "/home/nguyen-tuan/VNPT/CAM/sound_cam/own_dataset"

################### for Resnet18 , squeeze 20-2
TARGET_SAMPLE_RATE = 44100
NUM_SAMPLES = 44100  # 44100
N_FFT = 4096
HOP_LENGTH = 308
WIN_LENGTH = 2205
NORMALIZE = True
MIX_UP = 0.1
F_MAX = 18000
N_MELS = 224

#################### for Aug-Squeeze
# TARGET_SAMPLE_RATE = 44100
# NUM_SAMPLES = 44100  # 44100
# N_FFT = 1024
# # HOP_LENGTH = 512
# # WIN_LENGTH = 2205
# NORMALIZE = True
# # MIX_UP = 0.1
# # F_MAX = 18000
# N_MELS = 128

##################### for Test weights
# TARGET_SAMPLE_RATE = 16000
# NUM_SAMPLES = 16000
# N_FFT = 4096
# HOP_LENGTH = 512
# WIN_LENGTH = 2048
# NORMALIZE = True
# MIX_UP = 0.1
# N_MELS = 224
#####################

DEVICE = 'cuda'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
MOMENTUM = 0.9

PATH_TO_SAVE_MODEL = "/home/nguyen-tuan/VNPT/CAM/sound_cam/report/test_weights"

# transfer learning
# Resnet18
# NAME_MODEL = "Own_Aug_MFCC_Resnet18_22_2.pt"
# NAME_MODEL = "Own_Aug_MFCC_DB_Resnet18_22_2.pt"
NAME_MODEL = "Own_Aug_Mel_DB_Resnet18_22_2.pt" # *
# NAME_MODEL = "Own_Aug_Mel_Resnet18_22_2.pt"
# NAME_MODEL = "Own_Aug_Resnet18_14_2.pt"

# Squeeze11
# NAME_MODEL = "Own_Aug_MFCC_DB_Squeeze11_22_2_no.pt"
# NAME_MODEL = "Own_Aug_Mel_Squeeze11_22_2.pt"
# NAME_MODEL = "Own_Aug_Mel_DB_Squeeze11_22_2.pt"
# NAME_MODEL = "Own_Aug_MFCC_Squeeze11_22_2.pt"
# NAME_MODEL = "Own_Aug_MFCC_DB_Squeeze11_22_2.pt"

# No transfer
# Resnet18
# NAME_MODEL = "Own_Aug_MFCC_Resnet18_22_2_no.pt"
# NAME_MODEL = "Own_Aug_MFCC_DB_Resnet18_22_2_no.pt" # **
# NAME_MODEL = "Own_Aug_Mel_DB_Resnet18_22_2_no.pt"  # **********
# NAME_MODEL = "Own_Aug_Mel_Resnet18_22_2_no.pt"
# NAME_MODEL = "Own_Aug_Resnet18_14_2_no.pt"

# Squeeze11
# NAME_MODEL = "Own_Aug_MFCC_DB_Squeeze11_22_2_no.pt"
# NAME_MODEL = "Own_Aug_Mel_Squeeze11_22_2_no.pt"
# NAME_MODEL = "Own_Aug_Mel_DB_Squeeze11_22_2_no.pt"
# NAME_MODEL = "Own_Aug_MFCC_Squeeze11_22_2_no.pt"
# NAME_MODEL = "Own_Aug_MFCC_DB_Squeeze11_22_2_no.pt"

PATH_TO_MODEL = "/home/nguyen-tuan/VNPT/CAM/sound_cam/report/test_weights/{}".format(NAME_MODEL)  # foramat: Dataset_Model_Day_Month_ID_pretrained.pt
# PATH_TO_MODEL = "/home/nguyen-tuan/VNPT/CAM/sound_cam/weights/{}".format(NAME_MODEL)  # foramat: Dataset_Model_Day_Month_ID.pt

NAME_SAVE_MODEL = "Own_Aug_MFCC_DB_Resnet18_22_2_no.pt"

NAME_SAVE_PLOT = "Own_Aug_MFCC_DB_Resnet18_22_2_no.png"
PATH_TO_FIGURE = "/home/nguyen-tuan/VNPT/CAM/sound_cam/report/test_graphs/{}".format(NAME_SAVE_PLOT)

# CLASS_MAPPING = ["males", "females"]
# CLASS_MAPPING = ["cry", "laugh", "noise", "silence"]

# LABEL_DICT = {'dog': 0, 'rooster': 1, 'pig': 2, 'cow': 3, 'frog': 4, 'cat': 5, 'hen': 6, 'insects': 7, 'sheep': 8, 'crow': 9, 'rain': 10, 'sea_waves': 11, 'crackling_fire': 12, 'crickets': 13, 'chirping_birds': 14, 'water_drops': 15, 'wind': 16, 'pouring_water': 17, 'toilet_flush': 18, 'thunderstorm': 19, 'crying_baby': 20, 'sneezing': 21, 'clapping': 22, 'breathing': 23, 'coughing': 24, 'footsteps': 25, 'laughing': 26, 'brushing_teeth': 27, 'snoring': 28, 'drinking_sipping': 29, 'door_wood_knock': 30, 'mouse_click': 31, 'keyboard_typing': 32, 'door_wood_creaks': 33, 'can_opening': 34, 'washing_machine': 35, 'vacuum_cleaner': 36, 'clock_alarm': 37, 'clock_tick': 38, 'glass_breaking': 39, 'helicopter': 40, 'chainsaw': 41, 'siren': 42, 'car_horn': 43, 'engine': 44, 'train': 45, 'church_bells': 46, 'airplane': 47, 'fireworks': 48, 'hand_saw': 49}
# LABEL_DICT_CONVERT = {0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog', 5: 'cat', 6: 'hen', 7: 'insects', 8: 'sheep', 9: 'crow', 10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets', 14: 'chirping_birds', 15: 'water_drops', 16: 'wind', 17: 'pouring_water', 18: 'toilet_flush', 19: 'thunderstorm', 20: 'crying_baby', 21: 'sneezing', 22: 'clapping', 23: 'breathing', 24: 'coughing', 25: 'footsteps', 26: 'laughing', 27: 'brushing_teeth', 28: 'snoring', 29: 'drinking_sipping', 30: 'door_wood_knock', 31: 'mouse_click', 32: 'keyboard_typing', 33: 'door_wood_creaks', 34: 'can_opening', 35: 'washing_machine', 36: 'vacuum_cleaner', 37: 'clock_alarm', 38: 'clock_tick', 39: 'glass_breaking', 40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn', 44: 'engine', 45: 'train', 46: 'church_bells', 47: 'airplane', 48: 'fireworks', 49: 'hand_saw'}

LABEL_DICT = {'baby_cry': 0, 'other_sound': 1}
LABEL_DICT_CONVERT = {0: 'baby_cry', 1: 'other_sound'}

LABEL_DICT_CAL = {'baby_cry': 1, 'other_sound': 0}

# N_CLASSES = 50
N_CLASSES = 2