from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from models.loss_draw import LossHistory

args = parameter_parser()

"""
The graph feature and all the pattern feature are fed intto the fully-connected layer to get the weight parameter;
Then, the graph feature that multiples the graph weight merged with the pattern features multiple the pattern weight;
Finally, output the final prediction result and the weights of graph and patterns
"""


class EncoderWeight:
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

        graph2vec = tf.keras.layers.Dense(250, activation='relu', name='outputgraphvec')(input_dim)
        graphweight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputgraphweight')(graph2vec)
        newgraphvec = tf.keras.layers.Multiply(name='outputnewgraphvec')([graph2vec, graphweight])

        pattern1vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern1vec')(input_dim)
        pattern1weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern1weight')(pattern1vec)
        newpattern1vec = tf.keras.layers.Multiply(name='newpattern1vec')([pattern1vec, pattern1weight])

        pattern2vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern2vec')(input_dim)
        pattern2weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern2weight')(pattern2vec)
        newpattern2vec = tf.keras.layers.Multiply(name='newpattern2vec')([pattern2vec, pattern2weight])

        pattern3vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern3vec')(input_dim)
        pattern3weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern3weight')(pattern3vec)
        newpattern3vec = tf.keras.layers.Multiply(name='newpattern3vec')([pattern3vec, pattern3weight])

        mergevec = tf.keras.layers.Concatenate(name='mergevec')(
            [newgraphvec, newpattern1vec, newpattern2vec, newpattern3vec])
        mergevec = tf.keras.layers.Dense(100, activation='relu', name='outputmergevec')(mergevec)
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(mergevec)

        model = tf.keras.Model(inputs=[input_dim], outputs=[prediction])

        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model

    """
    Training model
    """
    def train(self):
        # 创建一个实例history
        # history = LossHistory()
        train_history = self.model.fit([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train],
                                       self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                       class_weight=self.class_weight, validation_split=0.2, verbose=2)
        print(str(train_history.history))
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

        # graphweight
        graphweight = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputgraphweight').output)
        graphweight_output = graphweight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        print(graphweight_output)

        # pattern1weight
        pattern1weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern1weight').output)
        pattern1weight_output = pattern1weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        print(pattern1weight_output)

        # pattern2weight
        pattern2weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern2weight').output)
        pattern2weight_output = pattern2weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        print(pattern2weight_output)

        # pattern3weight
        pattern3weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern3weight').output)
        pattern3weight_output = pattern3weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        print(pattern3weight_output)

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
