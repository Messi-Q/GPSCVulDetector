from __future__ import print_function
from parser import parameter_parser
from sklearn.utils import compute_class_weight
import tensorflow as tf

args = parameter_parser()

"""
Feed forward neural network
"""


class FFNN:
    def __init__(self, pattern_train, pattern_test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        super(FFNN, self).__init__()
        input = tf.keras.Input(12, name='input')
        self.pattern_train = pattern_train
        self.pattern_test = pattern_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight1 = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        self.class_weight2 = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_test)

        pattern_feature_input = tf.keras.layers.Dense(250, activation=tf.nn.relu)(input)
        pattern_feature_extract = tf.keras.layers.Dense(250, activation=tf.nn.relu, name='patternfeature')(pattern_feature_input)
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(pattern_feature_extract)
        model = tf.keras.Model(inputs=[input], outputs=[prediction])

        model.summary()
        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def pattern_train_feature_extract(self):
        self.model.fit(self.pattern_train, self.y_train, batch_size=self.batch_size,
                       epochs=self.epochs, class_weight=self.class_weight1)
        pattern_feature_train_layer = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('patternfeature').output)
        pattern_feature_train = pattern_feature_train_layer.predict(self.pattern_train)
        print(pattern_feature_train)

    def pattern_test_feature_extract(self):
        self.model.fit(self.pattern_test, self.y_test, batch_size=self.batch_size,
                       epochs=self.epochs, class_weight=self.class_weight2)
        pattern_feature_test_layer = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('patternfeature').output)
        pattern_feature_test = pattern_feature_test_layer.predict(self.pattern_test)
        print(pattern_feature_test)
