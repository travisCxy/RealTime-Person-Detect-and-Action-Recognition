import argparse
import torch

parser = argparse.ArgumentParser('ACTION RECOGNITION')

"------------------------------------ Gernal options ------------------------------------"
parser.add_argument('--dataset', default='UCF', type=str, help='Mark of dataset, human or hands')
parser.add_argument('--modality', default='rgb', type=str, help='modality, rgb|flow|depth|skeleton')
#parser.add_argument('--work', default='train', type=str, help='indicate which work will be done')
parser.add_argument('--method', default='global', type=str, help='method')
parser.add_argument('--idea', default='LSTM', type=str, help='idea to be proved')
parser.add_argument('--work', default='TV',type=str, help='train and valid')

"------------------------------------ Model options ------------------------------------"
parser.add_argument('--basicnet', default='inception', type=str, help='Type of basic net')
parser.add_argument('--res', default=False, type=bool, help='use res in encoder')
parser.add_argument('--encoder', default=True, type=bool, help='add encoder before net')
parser.add_argument('--num_classes', default=249, type=int, help='num of action class')
parser.add_argument('--local_size', default=72, type=int,help='size of local part')
parser.add_argument('--input_size', default=224, type=int,help='size of input')
parser.add_argument('--time_sample', default=32, type=int,help='size of time smaple')
parser.add_argument('--crop_size', default=64,type=int, help='crop size')

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
parser.add_argument('--use_cuda', default=1, type=int, help='use cuda')
parser.add_argument('--num_workers', default=8, help='num of workers')
parser.add_argument('--model_path', default='../model/iso_model/ucfbest0.96', type=str, help='path to pretrained model')
parser.add_argument('--fix', type=bool, default=False, help='fix some layer')

"------------------------------------ Data options ------------------------------------"
parser.add_argument('--datapath', default='../../dataset/SSD', help='path to dataset')
parser.add_argument('--listpath', default='data/ISO_GD/list', help='path tot list directory')
parser.add_argument('--trainList', default='train_list.txt', help='train list')
parser.add_argument('--validList', default='valid_list.txt', help='valid list')

"------------------------------------ VideoDemo options ------------------------------------"
parser.add_argument('--videopath', default='example/test_sample.avi', type=str, help='path to test vieo')
parser.add_argument('--save_video', default=True, type=bool, help='save video or not')
parser.add_argument('--outpath', default='example', type=str, help='path to output video')
parser.add_argument('--show', default=False, type=bool, help='show video')
parser.add_argument('--oneShot', default=False, type=bool, help='one video recognize one time')
parser.add_argument('--demo_mode', default='video', type=str, help='video demo or camera')
parser.add_argument('--camera', default=0, type=str, help='camera')
parser.add_argument('--step', default=1, type=int, help='sample step')
opt = parser.parse_args()
