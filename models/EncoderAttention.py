from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from models.loss_draw import LossHistory

args = parameter_parser()

"""
Add the attention mechanism for graph feature and pattern feature
"""


class EncoderAttention:
    def __init__(self, graph_train, graph_test, pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
                 pattern3test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        input_dim = tf.keras.Input(shape=(1, 250), name='input')

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.pattern1train = pattern1train
        self.pattern2train = pattern2train
        self.pattern3train = pattern3train
        self.pattern1test = pattern1test
        self.pattern2test = pattern2test
        self.pattern3test = pattern3test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)

        graph2vec = tf.keras.layers.Dense(250, activation='relu', name='graph2vec')(input_dim)
        pattern1vec = tf.keras.layers.Dense(250, activation='relu', name='pattern1vec')(input_dim)
        pattern2vec = tf.keras.layers.Dense(250, activation='relu', name='pattern2vec')(input_dim)
        pattern3vec = tf.keras.layers.Dense(250, activation='relu', name='pattern3vec')(input_dim)

        mergevec = tf.keras.layers.Concatenate(name='concatvec')([graph2vec, pattern1vec, pattern2vec, pattern3vec])
        mergevec = tf.keras.layers.Dense(250, activation='relu', name='mergevec')(mergevec)

        graphatt = tf.keras.layers.Attention(name='graphatt')([mergevec, graph2vec])
        patten1att = tf.keras.layers.Attention(name='patten1att')([mergevec, pattern1vec])
        patten2att = tf.keras.layers.Attention(name='patten2att')([mergevec, pattern2vec])
        patten3att = tf.keras.layers.Attention(name='patten3att')([mergevec, pattern3vec])

        mergeattvec = tf.keras.layers.Concatenate(name='concatvec2')([graphatt, patten1att, patten2att, patten3att])
        mergeattvec = tf.keras.layers.Dense(100, activation='relu', name='mergeattvec')(mergeattvec)

        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(mergeattvec)

        model = tf.keras.Model(inputs=[input_dim], outputs=[prediction])

        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model

    """
    Training model
    """
    def train(self):
        # create the history instance
        # history = LossHistory()
        self.model.fit([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train], self.y_train,
                       batch_size=self.batch_size, epochs=self.epochs, class_weight=self.class_weight,
                       validation_split=0.2, verbose=2)
        # self.model.save_weights("model.pkl")
        # history.loss_plot('epoch')

    """
    Testing model
    """
    def test(self):
        # self.model.load_weights("_model.pkl")
        values = self.model.evaluate([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                     self.y_test, batch_size=self.batch_size, verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])

        # predictions
        predictions = (self.model.predict([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                          batch_size=self.batch_size).round())

        predictions = predictions.flatten()
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
