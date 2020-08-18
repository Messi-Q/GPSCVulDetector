import numpy as np
from parser import parameter_parser
from models.EncoderConv1D import EncoderConv1D
from preprocessing import get_graph_feature, get_pattern_feature

args = parameter_parser()


def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test, label_by_extractor_train, label_by_extractor_valid = get_pattern_feature()

    graph_train = np.array(graph_train)  # The training set of graph feature
    graph_test = np.array(graph_test)  # The testing set of graph feature

    # The training set of patterns' feature
    pattern1train = []
    pattern2train = []
    pattern3train = []
    pattern4train = []
    for i in range(len(pattern_train)):
        pattern1train.append([pattern_train[i][0]])
        pattern2train.append([pattern_train[i][1]])
        pattern3train.append([pattern_train[i][2]])
        # pattern4train.append([pattern_train[i][3]])

    # The testing set of patterns' feature
    pattern1test = []
    pattern2test = []
    pattern3test = []
    pattern4test = []
    for i in range(len(pattern_test)):
        pattern1test.append([pattern_test[i][0]])
        pattern2test.append([pattern_test[i][1]])
        pattern3test.append([pattern_test[i][2]])
        # pattern4test.append([pattern_test[i][3]])

    pattern_train = np.array(pattern_train)
    pattern_test = np.array(pattern_test)

    # The ground truth label of certain contract in training set
    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    # The label of certain contract in testing set
    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    if args.model == 'EncoderConv1D':  # Conv layer and dense layer
        model = EncoderConv1D(graph_train, graph_test, pattern_train, pattern_test, y_train, y_test)

    model.train()  # training
    model.test()  # testing


if __name__ == "__main__":
    main()
