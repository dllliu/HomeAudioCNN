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
    raw = wave.open(param)
    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    # gets the frame rate
    f_rate = raw.getframerate()
    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
    # using matplotlib to plot
    # creates a new figure
    plt.figure(1)
    # title of the plot
    plt.title(param)
    # label of x-axis
    plt.xlabel("Time")
    # actual plotting
    plt.plot(time, signal)
    # shows the plot
    # in new window
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
         visualize_spec(sound)
         gen_mel_spec(sound)  

if __name__ == '__main__': main()
