import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import librosa
import os
from glob import glob
import numpy as np
import pandas as pd

SOURCE_DATA='Folded-AudioData/'
GENERATED_DATA='Spectrograms/'

file_ext = '*.wav'

child_dirs = os.listdir(SOURCE_DATA)

itr = iter(child_dirs)
fold1 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold2 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold3 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold4 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold5 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold6 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold7 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold8 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold9 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold10 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))

class_names = [fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10]
all_folds = ["fold0","fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold9"]

def create_specs(audio_file, image_file):
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    y, sr = librosa.load(audio_file)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)
    #plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    #plt.show()
    plt.savefig(image_file,bbox_inches='tight', pad_inches=0)
    plt.close()

def save_specs(input_path, output_path):
    for fn in glob(os.path.join(input_path,file_ext)):
        start = fn.find("\\") + 1
        name = fn[start:]
        input_file = os.path.join(input_path, name)
        output_file = os.path.join(output_path, name.replace('.wav', '.png'))
        create_specs(input_file, output_file)

def make_file_structure():
    for fold in all_folds:
        OutFolder = os.path.join(GENERATED_DATA,fold)
        if not os.path.exists(OutFolder):
            os.makedirs(OutFolder)
    

make_file_structure()
specs_dir = os.listdir(SOURCE_DATA)

for dir in specs_dir:
    input = os.path.join(SOURCE_DATA,dir)
    output = os.path.join(GENERATED_DATA,dir)
    save_specs(input,output)
