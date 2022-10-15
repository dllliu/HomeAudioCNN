import librosa
import os
from glob import glob
import numpy as np
import pandas as pd
import soundfile as sf

SOURCE_DATA='Folded-AudioData/'
GENERATED_DATA='GeneratedData/'

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

def extract_feature(file_name):
    print('Extracting', file_name)
    X, sample_rate = sf.read(file_name, dtype = 'float32')
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft,
    	sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    	sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
    
def parse_audio_files(parent_dir,sub_dir,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0) # 193 => total features
    for fn in glob(os.path.join(parent_dir, sub_dir, file_ext)):
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        arr = fn.split("-")
        class_label = arr[2]
        labels = np.append(labels, class_label)
    return np.array(features, dtype=np.float32), np.array(labels, dtype = np.int8)

sub_dirs = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])
                  
def saveFeatures():
    for sub_dir in child_dirs:
        features, labels = parse_audio_files(SOURCE_DATA,sub_dir)
        np.savez("{0}{1}".format(GENERATED_DATA, sub_dir), features=features,labels=labels)
    #numpy's savez(~) method writes multiple Numpy arrays to a single file in .npz format. 
    #Unlike the savez_compressed(~) method, savez(~) merely bundles up the arrays without compression
    
saveFeatures()
    