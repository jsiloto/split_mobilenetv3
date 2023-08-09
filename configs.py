import argparse
import os
import shutil

import yaml

from utils import mkdir_p


def load_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.full_load(f)

def parse_config(args):
    if args.config is not None:
        config = load_config(args.config)
        return config

    config = {}
    config['dataset'] = load_config(args.dataset)
    config['base_model'] = load_config(args.model)['base_model']
    config['model'] = load_config(args.model)['model']
    config['hyper'] = load_config(args.hyper)
    config_name = args.dataset.split("/")[-1].split(".")[0] + "_" \
                  + args.model.split("/")[-1].split(".")[0] + "_" \
                  + args.hyper.split("/")[-1].split(".")[0]

    config['checkpoint'] = os.path.join(args.project, config_name, args.name)

    if args.clean:
        print("Cleaning checkpoint folder")
        shutil.rmtree(config['checkpoint'])
    if not os.path.isdir(config['checkpoint']):
        mkdir_p(config['checkpoint'])

    return config

def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument('--config', help='Master yaml file path, overrides every other config')
    parser.add_argument('--dataset', default='./configs/dataset/stl10.yaml', help='yaml file path')
    parser.add_argument('--model', default='./configs/model/regular.yaml', help='yaml file path')
    parser.add_argument('--hyper', default='./configs/hyper/default.yaml', help='yaml file path')
    parser.add_argument('-p', '--project', default='./checkpoints', type=str, metavar='PATH',
                        help='checkpoint project path')
    parser.add_argument('-n', '--name', default='default', type=str, metavar='PATH',
                        help='checkpoint project path')
    parser.add_argument('--clean', action='store_true', help='clean checkpoint folder')

def get_config_from_args(parser: argparse.ArgumentParser):
    add_config_args(parser)
    args = parser.parse_args()
    return parse_config(args)



