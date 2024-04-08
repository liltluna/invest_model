import keras
import numpy as np
import pandas as pd
from config import *
from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from sklearn.metrics import classification_report, confusion_matrix


class CNN:
    def __init__(self, category_num=15):
        self.input_w = cnn_params['input_w']
        self.input_h = cnn_params['input_h']
        self.category_num = cnn_params['num_classes']
        self.model = self.build()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=Adadelta(),
                           metrics=['accuracy', 'mae', 'mse'])

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
            self.input_w, self.input_h, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.category_num, activation='softmax'))
        return model

    def reverseOneHot(self, predictions):
        reversed_x = []
        for x in predictions:
            reversed_x.append(np.argmax(np.array(x)))
        return reversed_x

    def train(self, training_df, test_df, params=cnn_params):
        print("Training is starting ...")

        train_images = training_df.iloc[:, 2:].to_numpy()
        train_labels = training_df.iloc[:, 0]
        train_prices = training_df.iloc[:, 1]

        test_images = test_df.iloc[:, 2:].to_numpy()
        test_labels = test_df.iloc[:, 0]
        test_prices = test_df.iloc[:, 1]

        test_labels = keras.utils.np_utils.to_categorical(
            test_labels, params["num_classes"])
        train_labels = keras.utils.np_utils.to_categorical(
            train_labels, params["num_classes"])

        train_images = train_images.reshape(
            train_images.shape[0], params["input_w"], params["input_h"], 1)
        test_images = test_images.reshape(
            test_images.shape[0], params["input_w"], params["input_h"], 1)

        # metrics.accuracy_score, metrics.recall_score, metrics.average_precision_score, metrics.confusion_matrix
        train_data_size = train_images.shape[0]
        test_data_size = test_images.shape[0]

        print("model will be trained with {} and be tested with {} sample".format(
            train_data_size, test_data_size))
        # fit the model to the training data
        print("Fitting model to the training data...")
        self.model.fit(train_images, train_labels,
                       batch_size=params["batch_size"], epochs=params["epochs"], verbose=1, validation_data=None)

        predictions = self.model.predict(
            test_images, batch_size=params["batch_size"], verbose=1)
        print(self.model.evaluate(test_images, test_labels,
                                  batch_size=params["batch_size"], verbose=1))

        print("Train conf matrix: \n", confusion_matrix(np.array(self.reverseOneHot(train_labels)),
                                                        np.array(self.reverseOneHot(self.model.predict(train_images, batch_size=params["batch_size"], verbose=1)))))

        print("Test conf matrix: \n",  confusion_matrix(np.array(self.reverseOneHot(test_labels)),
                                                        np.array(self.reverseOneHot(predictions))))

        return predictions, test_labels, test_prices

    def preProcess(self, train_path, test_path):
        train_df = pd.read_csv(train_path, header=None, index_col=None, delimiter=',')
        test_df = pd.read_csv(test_path, header=None, index_col=None, delimiter=',')

        l0_train = train_df.loc[train_df[0] == 0]
        l1_train = train_df.loc[train_df[0] == 1]
        l2_train = train_df.loc[train_df[0] == 2]

        l0_size = l0_train.shape[0]
        l1_size = l1_train.shape[0]
        l1_repeat_count = l0_size // l1_size
        l2_size = l2_train.shape[0]
        l2_repeat_count = l0_size // l2_size

        l1_new = [row for _ in range(l1_repeat_count) for row in l1_train.itertuples()]
        l2_new = [row for _ in range(l2_repeat_count) for row in l2_train.itertuples()]

        l1_new_df = pd.DataFrame(l1_new, columns=train_df.columns)
        l2_new_df = pd.DataFrame(l2_new, columns=train_df.columns)

        train_df = pd.concat([train_df, l1_new_df, l2_new_df])

        # 输出调整前后的样本数量比
        print("Before")
        print("l0_size:", l0_size, "l1_size:", l1_size, "l2_size:", l2_size)
        print("After appending new samples:")
        print("l0_size:", train_df[train_df[0] == 0].shape[0],
            "l1_size:", train_df[train_df[0] == 1].shape[0],
            "l2_size:", train_df[train_df[0] == 2].shape[0])