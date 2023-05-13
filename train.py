import argparse
import os
import yaml

from train_classifier import train_classifier

parser = argparse.ArgumentParser(description='Train All Models')
parser.add_argument('--dataset', required=True, help='dataset name')
parser.add_argument('--regular', help='regular model yaml file path')
parser.add_argument('--split', help='split model yaml file path')
parser.add_argument('--compression', help='compression model yaml file path')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

def load_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.full_load(f)

def main():
    args = parser.parse_args()

    if args.regular is not None:
        regular_config = load_config(args.regular)
        regular_config['checkpoint'] = os.path.join(args.checkpoint, args.dataset, "regular")
        train_classifier(args.dataset, regular_config)

    if args.split is not None:
        split_config = load_config(args.split)
        split_config['checkpoint'] = os.path.join(args.checkpoint, args.dataset, "split")
        train_classifier(args.dataset, split_config)


if __name__ == '__main__':
    main()
