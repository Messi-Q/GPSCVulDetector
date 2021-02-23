from __future__ import print_function
from parser import parameter_parser
from sklearn.utils import compute_class_weight
import tensorflow as tf

args = parameter_parser()

"""
Feed forward neural network
"""


class FFNN:
    def __init__(self, pattern1, pattern2, pattern3, label, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        super(FFNN, self).__init__()
        input = tf.keras.Input(shape=(1, 4), name='input')
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.pattern3 = pattern3
        self.label = label
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=label)

        pattern1 = tf.keras.layers.Dense(250, activation=tf.nn.relu)(input)
        pattern2 = tf.keras.layers.Dense(250, activation=tf.nn.relu)(input)
        pattern3 = tf.keras.layers.Dense(250, activation=tf.nn.relu)(input)

        mergevec = tf.keras.layers.Concatenate()([pattern1, pattern2, pattern3])
        Dense = tf.keras.layers.Dense(250, activation='relu', name='patternfeature')(mergevec)
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(Dense)

        model = tf.keras.Model(inputs=[input], outputs=[prediction])

        model.summary()
        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self):
        self.model.fit([self.pattern1, self.pattern2, self.pattern3], self.label, batch_size=self.batch_size,
                       epochs=self.epochs, class_weight=self.class_weight)
        # self.model.save_weights("model.pkl")
        # decoder the training vectors
        pattern_feature = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('patternfeature').output)
        print(pattern_feature)
