import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import KFold

### Define feedforward network architecture ###
def get_network():
    input_shape = (193,)
    num_classes = 10
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    return model


### Train and evaluate via 10-Folds cross-validation ###
accuracies = []
folds = np.array(['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10'])
load_dir = "GeneratedData/"
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds):
    x_train, y_train = [], []
    for ind in train_index:
        data = np.load("{0}/{1}.npz".format(load_dir,folds[ind]))
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
    model.fit(x_train, y_train, epochs = 50, batch_size = 24, verbose = 0)
    l, a = model.evaluate(x_test, y_test, verbose = 0)
    accuracies.append(a)
    print("Loss: {0} | Accuracy: {1}".format(l, a))

print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))