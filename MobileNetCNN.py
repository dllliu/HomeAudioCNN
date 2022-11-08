import numpy as np
import os
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import pandas as pd

import keras
import tensorflow as tf

SOURCE_DATA='Folded-AudioData/'
GENERATED_DATA='Spectrograms/'
file_ext = '*.wav'

def load_images(file):
    images = []
    image = (keras.utils.load_img(file, target_size=(224, 224, 3)))
    array = tf.keras.preprocessing.image.img_to_array(image)
    images.append(array)
    return images

def extract(path):
    feature = []
    label = []
    for file in glob(path):
        arr = file.split("-")
        fn_label = str(arr[1])[0]
        images = load_images(file)
        feature += images
        label += fn_label
    return feature,label

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
import seaborn as sns

#Extract Relevent Features From Specs with MobileNet CNN
def base_model_mobile():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers: #freeze all layers
        layer.trainable = False
    return model
    
x_train = []
x_test = []
y_train = []
y_test = []

from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

folds = np.array(['fold0','fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9'])
load_dir = "Spectrograms/"
accuracies = []
losses = []
acc = []
val_acc = []
lo = []
val_lo = []
kf = KFold(n_splits=10)
#skf = StratifiedKFold(n_splits=10)
count = 0
for train_index, test_index in kf.split(folds): #Splits into training and testing sets
    x_train, y_train = [], []
    for ind in train_index:
        param = os.path.join(load_dir,'fold'+str(ind),"*png")
        features,labels = extract(param)
        x_train += features
        y_train += labels
    test_param = os.path.join(load_dir,'fold'+str(test_index[0]),"*png")
    test_features, test_labels = extract(test_param)
    x_test += test_features
    y_test += test_labels

    x_train_norm = np.array(x_train) / 255
    x_test_norm = np.array(x_test) / 255

    #one hot encode
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    model = base_model_mobile()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train_norm,y_train_encoded,validation_data=(x_test_norm, y_test_encoded),batch_size=10,epochs=5) 

    #determine which layer to slice at for freeze/unfreeze

    ###
    for i, layer in enumerate(model.layers):
       print(i, layer.name)
    ###

    for layer in model.layers[:67]:
       layer.trainable = False

    for layer in model.layers[67:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy',metrics=['accuracy'])
    #model.summary()
    history = model.fit(x_train_norm,y_train_encoded, validation_data= (x_test_norm, y_test_encoded), batch_size=10,epochs=15) #epochs = 15
    
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
    acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])
    #plt.savefig("Accuracy_Graph_"+str(count)+"_.png")
    #plt.close()

    # summarize history for loss
    
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
    lo.append(history.history['loss'])
    val_lo.append(history.history['val_loss'])
    #plt.savefig("Loss_Graph_"+str(count)+"_.png")
    #plt.close()
    
    #model.save("Model" + str(count) + ".h5")

    l, a = model.evaluate(x_test_norm,y_test_encoded,verbose = 0)
    accuracies.append(a)
    losses.append(l)
    print(accuracies)
    print(losses)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    y_predicted = model.predict(x_test_norm)
    con_mat_df = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df,annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Confusion_Matrix_CNN"+str(count)+".png")
    #plt.show()
    plt.close()
    count += 1

    print("Displaying Classification Report")
    classes = ["0-Voices","1-Locomotion","2-Digestive","3-Elements","4-Animals","5-Cook_App","6-Clean-App","7-Vent_App","8-Furniture","9-Instruments"]
    print(classification_report(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1), target_names=classes))
    x_train.clear()
    x_test.clear()
    y_train.clear()
    y_test.clear()

print(acc)
print(val_acc)
print(lo)
print(val_lo)

print("Average 10 Folds Accuracy:" + str((np.mean(accuracies))))
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(all_folds, accuracies, color ='maroon',
        width = 0.4)

plt.xlabel("Fold No")
plt.ylabel("Accuracy")
plt.title("Accuracy of Each Fold")
plt.show()
plt.close()
#plt.savefig(os.path.join(MODEL_GRAPHS,"Dataset_All_Folds_Accuracy" + ".png"))
