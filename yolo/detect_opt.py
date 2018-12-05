import argparse

parser = argparse.ArgumentParser()
"---------------------------------- general options ----------------------------------"
parser.add_argument('--dataset', type=str, default='coco', help='dataset')
parser.add_argument('--cfg', type=str, default='cfg/person.cfg', help='cfg file path')
parser.add_argument('--class_path', type=str, default='data/person.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
parser.add_argument('--img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('--data_config_path', type=str, default='cfg/coco.data', help='data config file path')
parser.add_argument('--show', type=bool, default=True)

"---------------------------------- detect options ----------------------------------"
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to images')
parser.add_argument('--output_folder', type=str, default='output', help='path to outputs')
parser.add_argument('--plot_flag', type=bool, default=True)
parser.add_argument('--txt_out', type=bool, default=False)
parser.add_argument('--detect_batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--detect_weights_path', type=str, default='weights/yolov3_person.pt')

"--------------------------------- train options  ------------------------------------"
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--train_batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--resume', default=False, help='resume training flag')

"-------------------------------- test options ---------------------------------------"
parser.add_argument('--test_batch_size', type=int, default=32, help='size of each image batch')
parser.add_argument('--test_weights_path', type=str, default='weights/yolov3-cup_50000.weights', help='path to weights file')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
