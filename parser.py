import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Vulnerability Detection')
    parser.add_argument('-D', '--dataset',type=str,default='reentrancy_DDCD',choices=['reentrancy_none',
                                        'reentrancy_DD','reentrancy_DDCD',])
    parser.add_argument('-M', '--model', type=str, default='ConvMHSA',
                        choices=['ConvMHSA'])
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--vector_dim', type=int, default=300, help='dimensions of vector')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, choices=['32','64','128'],help='batch size')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='threshold')

    return parser.parse_args()
