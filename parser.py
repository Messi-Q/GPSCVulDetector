import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Vulnerability Detection')

    parser.add_argument('-D', '--dataset', type=str, default='', choices=[])
    parser.add_argument('-M', '--model', type=str, default='CGE',
                        choices=['CGE', 'CGEVariant'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')

    return parser.parse_args()
