import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from numpy import reshape
from sklearn.metrics import classification_report


MODEL_GRAPHS= 'GeneratedGraphs' 
# Define network architecture #
def get_network():
    input_shape = (193,)
    num_classes = 10
    keras.backend.clear_session()
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
        optimizer='rmsprop', 
        metrics=['accuracy'])
        
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    
    model.compile(optimizer='rmsprop',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])
    model.summary()
    """
    return model


#Train and evaluate via 10-Folds cross-validation 
accuracies = []
losses = []
folds = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])
load_dir = "GeneratedData/"
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds): #Splits into training and testing sets
    x_train, y_train = [], []
    for ind in train_index:
        data = np.load("{0}/{1}.npz".format(load_dir,folds[ind]))
        #print(folds[ind]) all folds except for one
        x_train.append(data["features"])
        y_train.append(data["labels"])
    x_train = np.concatenate(x_train, axis = 0)
    y_train = np.concatenate(y_train, axis = 0)

    data = np.load("{0}/{1}.npz".format(load_dir,folds[test_index][0]))
    x_test = data["features"]
    y_test = data["labels"]
    
    # Possibly do mean normalization here on x_train and
    # x_test but using only x_train's mean and std.

    model = get_network()
    
    history = model.fit(x_train, y_train,validation_split=0.2, epochs = 70, batch_size = 24, verbose = 0) #epochs = 70
        
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    
    l, a = model.evaluate(x_test, y_test, verbose = 0)
    
    all_folds = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
    losses.append(l)
    accuracies.append(a)

    print("Loss: {0} | Accuracy: {1}".format(l, a))

print(model.summary())

y_pred = (model.predict(x_test) > 0.5).astype("int32")
print(y_test)
print(y_test.shape)

y_pred = np.argmax(model.predict(x_test),axis=1)
print(y_pred)
print(y_pred.shape)
print('Confusion Matrix')
con_mat_df = confusion_matrix(y_test, y_pred)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df,annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(MODEL_GRAPHS,"Confusion_Matrix" + ".png"))
#plt.show()

print("Displaying Classification Report")
print(classification_report(y_test, y_pred, target_names=all_folds))

print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(all_folds, accuracies, color ='maroon',
        width = 0.4)
 
plt.xlabel("Fold No")
plt.ylabel("Accuracy")
plt.title("Accuracy of Each Fold")
plt.savefig(os.path.join(MODEL_GRAPHS,"Dataset_All_Folds_Accuracy" + ".png"))
#plt.show()

#model.save("trained_model.h5")