import os
import re
import argparse
import mlflow
from train_net import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--entry_point', type=str, help='entrypoint name [ train, validate, predict ]')
parser.add_argument('--config_file', type=str, help='config file path')
parser.add_argument('--num_gpu', type=int, help='GPUs')
parser.add_argument('--pred_list', type=str, help='list of images to predict')
parser.add_argument('--weight_file',type=str, help='weight path')
parser.add_argument('--input',type=str, help='list input files')
parser.add_argument('--eval-only',help='evaluate a model performance')
# print( os.environ )

# def read_validate_result(text : str):
#     res = Trainer.test_with_TTA
#     matcher = re.compile(r'class_id\s=\s(\d+),\sname\s=\s([^,]+),\sap\s=\s(\d+.\d+%)\s+\(TP\s=\s(\d+),\sFP\s=\s(\d+)\)')
#     res = matcher.findall(text)

    # return res

def train(config_file: str, num_gpu: int):
    os.system(f"./train_net.py --config-file {config_file} --num-gpus {num_gpu}")
    validate(config_file)

def validate(config_file: str, weight_file):
    os.system(f"./train_net.py {config_file} --eval-only {weight_file}")
    # text = ''

    # with open('result.txt') as f:
    #     text = f.read().strip()
    
    # res = read_validate_result(text)

def predict(config_file: str, input: str, weight_file: str):
    os.system(f"./demo.py --config-file {config_file} --input {input} --opts MODEL.WEIGHT {weight_file}")
    #log_artifact( 'result.json', 'file' )

    
if __name__ == '__main__':
    print('test')
    args = parser.parse_args()
    # print(f'entry: {args.entry_point}')
    args.entry_point = "train"
    if args.entry_point == 'train':
        train(args.config_file, args.num_gpu)
    elif args.entry_point == 'validate':
        validate(args.config_file)
    elif args.entry_point == 'predict':
        predict(args.config_file, args.pred_list)

