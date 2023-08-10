import argparse
import json
import os
import shutil

import yaml

from configs import get_config_from_args
from train_classifier import train_classifier
from utils import mkdir_p

def main():
    parser = argparse.ArgumentParser(description='Train Model')
    configs = get_config_from_args(parser)
    with open(os.path.join(configs['checkpoint'], 'metadata.json'), "w") as f:
        json.dump(configs, f)
    train_classifier(configs)

if __name__ == '__main__':
    main()
