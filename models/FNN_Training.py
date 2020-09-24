from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

"""
Feed forward neural network
"""


class FNN:
    def __init__(self, in_dim, n_hidden, out_dim):
        super(FNN, self).__init__()



    def train(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
