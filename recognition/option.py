import argparse
import torch

parser = argparse.ArgumentParser('ACTION RECOGNITION')

"------------------------------------ Gernal options ------------------------------------"
parser.add_argument('--dataset', default='NTU', type=str, help='Mark of dataset, human or hands')
parser.add_argument('--work', default='TV',type=str, help='train and valid')

"------------------------------------ Model options ------------------------------------"
parser.add_argument('--basicnet', default='inception', type=str, help='Type of basic net')
parser.add_argument('--num_classes', default=249, type=int, help='num of action class')
parser.add_argument('--input_size', default=224, type=int,help='size of input')
parser.add_argument('--time_sample', default=32, type=int,help='size of time smaple')

"-------------------------------- Hyperparameters options ---------------------------------"
parser.add_argument('--LR', default=0.001, type=float, help='Learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--optim', default='SGD', type=str, help='optimizer')
parser.add_argument('--dropout_prob', default=0.5, type=float, help='dropout prob')

"------------------------------------ Train options ------------------------------------"
parser.add_argument('--num_epochs', default=10, type=int, help='num of epochs')
parser.add_argument('--trainBatch', default=8, type=int, help='train-batch-size')
parser.add_argument('--validBatch', default=4, type=int, help='valid-batch-size')
parser.add_argument('--print_step', default=200, type=int, help='step to print info')
parser.add_argument('--note', type=str, help='indicate train info')
parser.add_argument('--data_type', default='float', type=str, help='data type')
parser.add_argument('--num_workers', default=8, help='num of workers')
parser.add_argument('--model_path', default='../model/iso_model/ucfbest0.96', type=str, help='path to pretrained model')

"------------------------------------ Data options ------------------------------------"
parser.add_argument('--datapath', default='../../dataset/SSD', help='path to dataset')
parser.add_argument('--listpath', default='data/ISO_GD/list', help='path tot list directory')
parser.add_argument('--trainList', default='train_list.txt', help='train list')
parser.add_argument('--validList', default='valid_list.txt', help='valid list')

opt = parser.parse_args()
