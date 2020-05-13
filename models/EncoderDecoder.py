from __future__ import print_function
from parser import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from models.loss_draw import LossHistory

args = parameter_parser()


