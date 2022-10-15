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

child_dirs = os.listdir(GENERATED_DATA)

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

def create_specs(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    #plt.show()
    fig.savefig(image_file)
    plt.close(fig)
    
    
def save_specs(input_path, output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for fn in glob(os.path.join(input_path,file_ext)):
        start = fn.find("\\") + 1
        name = fn[start:]
        input_file = os.path.join(input_path, name)
        output_file = os.path.join(output_path, name.replace('.wav', '.png'))
        create_specs(input_file, output_file)
            
#child_dirs = os.listdir(SOURCE_DATA)
import keras
import tensorflow as tf

def load_images(file):
    images = []
    image = (keras.utils.load_img(file, target_size=(224, 224, 3)))
    array = tf.keras.preprocessing.image.img_to_array(image)
    images.append(array)
    return images

features = []
labels = []

"""
for dir in child_dirs:
    input = os.path.join(SOURCE_DATA,dir)
    output = os.path.join(GENERATED_DATA,dir)
    save_specs(input,output)
"""


for dir in child_dirs:
    for file in glob(os.path.join(GENERATED_DATA, dir, '*.png')):
        fn_label = file.split("-")[1]
        images = load_images(file)
        features += images
        labels += fn_label

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

folds = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])

x_train, y_train = [], []
x_train, x_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


#Mobile Net Implementation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))

train_features = base_model.predict(x_train_norm)
test_features = base_model.predict(x_test_norm)

model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=10, epochs=10)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set()

y_predicted = model.predict(test_features)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = [0,1,2,3,4,5,6,7,8,9]

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()



        