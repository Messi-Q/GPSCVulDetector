import numpy as np
from parser import parameter_parser
from models.EncoderConv1D import EncoderConv1D
from preprocessing import get_graph_feature, get_pattern_feature

args = parameter_parser()


def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test = get_pattern_feature()

    graph_train = np.array(graph_train)
    graph_test = np.array(graph_test)
    pattern_train = np.array(pattern_train)
    pattern_test = np.array(pattern_test)

    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)

    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)

    if args.model == 'EncoderConv1D':
        model = EncoderConv1D(graph_train, graph_test, pattern_train, pattern_test, y_train, y_test)
    elif args.model == 'EncoderMM':
        model = EncoderConv1D(graph_train, graph_test, pattern_train, pattern_test, y_train, y_test)
    model.train()
    model.test()


if __name__ == "__main__":
    main()
