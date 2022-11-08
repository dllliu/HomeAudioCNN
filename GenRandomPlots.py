import matplotlib.pylab as plt
from matplotlib.pyplot import specgram
from glob import glob
import librosa
import os
import random
import wave,sys
import numpy as np
import librosa.display

SOURCE_DATA='Folded-AudioData'
MODEL_GRAPHS= 'GeneratedGraphs' 

file_ext = '*.wav'

sub_dir = os.listdir(SOURCE_DATA)
child_dirs = os.listdir(SOURCE_DATA)

itr = iter(child_dirs)
fold0 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold1 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold2 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold3 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold4 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold5 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold6 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold7 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold8 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))
fold9 = glob(os.path.join(SOURCE_DATA,next(itr),file_ext))

class_names = [fold0,fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9]

random_sounds = []
for x in class_names:
    random_sounds.append(random.choice(x))
#print(random_sounds)
    
#Array of Values from Sound. Sound can be expressed as vibrations in the air caused by the oscillation of a speaker, voltage. 
#Computer interprets data like amplitude from the osccillation 


def visualize(param):
    # reading the audio file
    y,sr=librosa.load(param) #load the file
    plt.title(param)
    librosa.display.waveshow(y,sr=sr, x_axis='time', color='cyan')
    plt.savefig(os.path.join(MODEL_GRAPHS,'Waveplot' + str(param.split("\\")[2])+ ".png"))
    plt.close()
    #plt.show()
    
def visualize_spec(param):
    y,sr = librosa.load(param)
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(np.abs(D),ref = np.max)
    fig,ax = plt.subplots(figsize = (10,5))
    img = librosa.display.specshow(S,x_axis='time',y_axis='log',ax=ax)
    ax.set_title("Spectrogram: " + param,fontsize=10)
    fig.colorbar(img,ax=ax)
    plt.savefig(os.path.join(MODEL_GRAPHS,'Spectrogram' + str(param.split("\\")[2])+ ".png"))
    plt.close()
    #plt.show()
    
def gen_mel_spec(param):
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    y, sr = librosa.load(param)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title("Log-Mel-Spectrogram: " + param,fontsize=10)
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(os.path.join(MODEL_GRAPHS,'Log_Mel_Spectrogram' + str(param.split("\\")[2])+ ".png"))
    plt.close()

def main():
    for sound in random_sounds:
         visualize(sound)  
    for sound in random_sounds:
        visualize_spec(sound)
    for sound in random_sounds:
        gen_mel_spec(sound)

if __name__ == '__main__': main()
