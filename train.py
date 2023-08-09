import argparse
import os
import yaml

from configs import get_config_from_args
from train_classifier import train_classifier

def main():
    parser = argparse.ArgumentParser(description='Train Model')
    configs = get_config_from_args(parser)
    train_classifier(configs)

if __name__ == '__main__':
    main()
