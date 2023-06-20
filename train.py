import argparse
import os
import yaml

from train_classifier import train_classifier

parser = argparse.ArgumentParser(description='Train All Models')
parser.add_argument('--dataset', required=True, help='dataset name')
parser.add_argument('--model', help='regular model yaml file path')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

def load_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.full_load(f)

def main():
    args = parser.parse_args()

    if args.model is not None:
        model_name = args.model.split("/")[-1].split(".")[0]
        regular_config = load_config(args.model)
        regular_config['checkpoint'] = os.path.join(args.checkpoint, args.dataset, model_name)
        train_classifier(args.dataset, regular_config)

if __name__ == '__main__':
    main()
