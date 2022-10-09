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
GENERATED_DATA='GeneratedData'

file_ext = '*.wav'

sub_dir = os.listdir(SOURCE_DATA)
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
    plt.show()
    # can save with plt.savefig('filename')
    
def visualize_spec(param):
    y,sr = librosa.load(param)
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(np.abs(D),ref = np.max)
    fig,ax = plt.subplots(figsize = (10,5))
    img = librosa.display.specshow(S,x_axis='time',y_axis='log',ax=ax)
    ax.set_title("Spectogram: " + param,fontsize=10)
    fig.colorbar(img,ax=ax)
    plt.show()
    
def gen_mel_spec(param):
    y,sr = librosa.load(param)
    S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the mel spectogram
    img = librosa.display.specshow(S_db_mel,
                          x_axis='time',
                          y_axis='log',
                          ax=ax)
    ax.set_title('Mel Spectogram:' + param, fontsize=8)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()

def main():
    for sound in random_sounds:
         visualize(sound)  
         #visualize_spec(sound)
         #gen_mel_spec(sound)  

if __name__ == '__main__': main()
