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
"""
specs_dir = os.listdir(SOURCE_DATA)

for dir in specs_dir:
    input = os.path.join(SOURCE_DATA,dir)
    output = os.path.join(GENERATED_DATA,dir)
    save_specs(input,output)
"""    

def extract(path):
    feature = []
    label = []
    for file in glob(path):
        fn_label = file.split("-")[1]
        images = load_images(file)
        feature += images
        label += fn_label
    return feature,label
    
x_train = []
x_test = []
y_train = []
y_test = []

for file in glob("Spectrograms\\fold1\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold1\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold2\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold2\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold3\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold3\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold4\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold4\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold5\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold5\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold6\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold6\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold7\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold7\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold8\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold8\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold9\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold9\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y
    
for file in glob("Spectrograms\\fold10\\test\\*png"):
    x,y = extract(file)
    x_test += x
    y_test += y
    
for file in glob("Spectrograms\\fold10\\train\\*png"):
    x,y = extract(file)
    x_train += x
    y_train += y

#print(len(x_train))
#print(len(x_test))
#print(len(y_train))
#print(len(y_test))


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

#one hot encode
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

#Extract Relevent Features From Specs with MobileNet CNN

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
import seaborn as sns

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#train_features = base_model.predict(x_train_norm)
#test_features = base_model.predict(x_test_norm)

for layer in model.layers: #freeze all layers
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train_norm,y_train_encoded,validation_data=(x_test_norm, y_test_encoded),batch_size=15,epochs=10)

for i, layer in enumerate(model.layers):
   print(i, layer.name)


for layer in model.layers[:67]: #54
   layer.trainable = False
for layer in model.layers[67:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(x_train_norm,y_train_encoded, validation_data= (x_test_norm, y_test_encoded), batch_size=10,epochs=50)

#increase epochs and maybe increase layer cutoff
######################
"""
def get_network():
    input_shape = (224, 224, 3)
    num_classes = 10
    keras.backend.clear_session()
    model = Sequential()
    model.add(Flatten(input_shape=train_features.shape[1:]))
    model.add(Dense(1024, activation='relu')) 
    model.add(Dense(10, activation='softmax')) 
    
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=0.0001), metrics=['accuracy'])
    return model

model = get_network()

model.summary()

history = model.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=10, epochs=50)
"""
###################

model.save('TrainedModel.h5')

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
from sklearn.metrics import classification_report

y_predicted = model.predict(x_test_norm)

#y_predicted = model.predict(test_features)
con_mat_df = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df,annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig("Confusion_Matrix_CNN")

print("Displaying Classification Report")
classes = ["0-Voices","1-Locomotion","2-Digestive","3-Hygiene","4-Animals","5-Cook_App","6-Clean-App","7-Vent_App","8-Furniture","9-Instruments"]
print(classification_report(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1), target_names=classes))



