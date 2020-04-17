from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from generate_input import get_graph_feature, get_pattern_feature

np.random.seed(1)
tf.random.set_seed(1)
lr = 0.005


def multi_input_model(graph_train, pattern_train):
    """构建多输入模型"""
    input1 = tf.keras.Input(shape=(1, 250), name='input1')
    input2 = tf.keras.Input(shape=(3, 250), name='input2')

    graph_train = np.array(graph_train)
    pattern_train = np.array(pattern_train)

    graph_train = tf.keras.layers.Conv1D(100, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(input1)
    graph_train = tf.keras.layers.MaxPool1D(pool_size=1, strides=1)(graph_train)

    pattern_train = tf.keras.layers.Conv1D(100, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same')(input2)
    pattern_train = tf.keras.layers.MaxPool1D(pool_size=3, strides=3)(pattern_train)

    x = tf.keras.layers.concatenate([graph_train, pattern_train])
    # x = tf.keras.layers.Flatten(x)

    x = tf.keras.layers.Dense(50, activation='relu')(x)
    prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=[prediction])
    model.summary()

    return model


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    return reconstruction_error


if __name__ == '__main__':
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test, pattern_extractor_train, pattern_extractor_test = get_pattern_feature()
    model = multi_input_model(graph_train, pattern_train)
    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    adama = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    model.fit([np.array(graph_train), np.array(pattern_train)], y_train, epochs=40, batch_size=16,
              class_weight=class_weight)
    values = model.evaluate([np.array(graph_test), np.array(pattern_test)], y_test, batch_size=16)
    print("Accuracy: ", values[0])
    predictions = (model.predict([np.array(graph_test), np.array(pattern_test)], batch_size=16).round())
    predictions = predictions.flatten()

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print('False positive rate(FPR): ', fp / (fp + tn))
    print('False negative rate(FN): ', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('Recall(TPR): ', recall)
    precision = tp / (tp + fp)
    print('Precision: ', precision)
    print('F1 score: ', (2 * precision * recall) / (precision + recall))
