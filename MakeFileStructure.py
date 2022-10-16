import csv
import os
import wave
import time
import multiprocessing
from threading import Thread
import os
import numpy as np
import matplotlib
import pyaudio
import librosa
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix

Type="GENERATE"  #For loading our helper functions we tell the sytem that we are planning to Generate images
from subprocess import Popen # We need a specific library for running the images generation in parrallel
exec(open("helperFunctions.py","rb").read())

SOURCE_DATA='Folded-AudioData/'
GENERATED_DATA='Spectrograms/'

ToDoList=[]
SourceFoldersLabels = [f.path for f in os.scandir(SOURCE_DATA) if f.is_dir()]
for path in SourceFoldersLabels:
    FileList = np.array([f.path for f in os.scandir(path) if f.is_file() and (f.name.endswith(".wav"))])
    Label = os.path.split(path)[-1]
    OutFolderTrain = os.path.join(GENERATED_DATA,Label,'train')
    OutFolderTest = os.path.join(GENERATED_DATA,Label,'test')
    if not os.path.exists(OutFolderTrain):
        os.makedirs(OutFolderTrain)
    if not os.path.exists(OutFolderTest):
        os.makedirs(OutFolderTest)
    np.random.shuffle(FileList)
    trainCount =np.int(np.floor(0.8*FileList.shape[0]))
    train_set = FileList[:trainCount]
    test_set = FileList[trainCount:]
    for f in train_set:
        ToDoList.append((os.path.abspath(f),os.path.abspath(OutFolderTrain)))
    for f in test_set:
        ToDoList.append((os.path.abspath(f),os.path.abspath(OutFolderTest)))
    print("Finished class",Label,". Going to the next.")
Commands = [[sys.executable, "helperFunctions.py",t[0],t[1]] for t in ToDoList]
print("Done Creating our ToDoList. I'll start computing now, hold on.")
tempArray=[]
for i in range(len(Commands)):
    tempArray.append(Commands[i])
    if(len(tempArray)>=12):  ## <= To optimize you can type in here how many CPU cores/threads you have
        procs = [Popen(j) for j in tempArray ]
        for p in procs:
            p.wait()
        tempArray=[]
procs = [ Popen(j) for j in tempArray ]
for p in procs:
    p.communicate()
print("All Done.")
