import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stanfordcars',help="imagenetr , domainnet or cifar100_vit")
    parser.add_argument('--config', type=str, default='./exps/stanfordcars.json',
                        help='Json file of settings.')
    parser.add_argument('--topk', type=int, default=5)
    return parser


if __name__ == '__main__':
    main()
