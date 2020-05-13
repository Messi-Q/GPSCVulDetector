from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.utils import compute_class_weight
from preprocessing import get_graph_feature, get_pattern_feature

lr = 0.002
dropout = 0.2


def TrainDetector(inputvec):
    input = tf.keras.Input(shape=(4, 50), name='input')

    vecvalue = inputvec.numpy()

    vecvalue = tf.keras.layers.Conv1D(50, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(input)
    outputvec = tf.keras.layers.MaxPool1D(pool_size=4, strides=4)(vecvalue)

    outputvec = tf.keras.layers.Dense(20, activation=tf.nn.relu)(outputvec)
    prediction = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(outputvec)

    # x = tf.keras.layers.Flatten(x)
    model = tf.keras.Model(inputs=input, outputs=[prediction])
    model.summary()
    return model


def Encoder(graph_train, pattern1=None, pattern2=None, pattern3=None, pattern4=None):
    # self.graph2vec = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
    # self.pattern2vec = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
    # graph_train = self.graph2vec(graph_train)

    # pattern1 = self.pattern2vec(pattern1)
    # pattern2 = self.pattern2vec(pattern2)
    # pattern3 = self.pattern2vec(pattern3)

    convec = np.hstack([graph_train, pattern1, pattern2, pattern3])  # 4 (1338, 1, 250) -> (1338, 4, 250)
    merge2vec = tf.keras.layers.Dense(50, activation=tf.nn.relu)  # (1338, 4, 250) -> (1338, 4, 50)
    outputvec = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid)  # (1338, 4, 50) -> (1338, 4, 50)
    mergevec = merge2vec(convec)
    outputvec = outputvec(mergevec)
    return outputvec


def Decoder(code, graph_train, pattern1=None, pattern2=None, pattern3=None, pattern4=None):
    merge2vec = tf.keras.layers.Dense(50, activation=tf.nn.relu)  # (1338, 1, 100) -> (1338, 1, 100)
    outputvec = tf.keras.layers.Dense(250, activation=tf.nn.sigmoid)  # (1338, 1, 100) -> (1338, 1, 1000)
    att = tf.keras.layers.Attention()

    mergevec = merge2vec(code)
    originalvec = outputvec(mergevec)
    vecvalue = originalvec.numpy()

    value = np.hsplit(vecvalue, 4)
    # get the vector of decoder
    graph_train_decoder, pattern1_decoder, pattern2_decoder, pattern3_decoder = value[0], value[1], value[2], value[3]

    # TODO: 输出权重
    enc4 = att([graph_train_decoder, graph_train])

    return enc4


if __name__ == '__main__':
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test = get_pattern_feature()

    pattern1 = []
    pattern2 = []
    pattern3 = []
    pattern4 = []
    for i in range(len(pattern_train)):
        pattern1.append([pattern_train[i][0]])
        pattern2.append([pattern_train[i][1]])
        pattern3.append([pattern_train[i][2]])

    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    output = Encoder(np.array(graph_train), np.array(pattern1), np.array(pattern2), np.array(pattern3))

    # model = TrainDetector(output)
    # adama = tf.keras.optimizers.Adam(lr)
    # model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
    # class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    # model.fit(output, y_train, epochs=40, batch_size=16, class_weight=class_weight)

    out = Decoder(output, np.array(graph_train), np.array(pattern1), np.array(pattern2), np.array(pattern3))
    print(out)

