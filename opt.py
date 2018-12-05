import argparse

parser = argparse.ArgumentParser('Real Time Action Recognition')

"---------------------    Gernal Options   ---------------------"
parser.add_argument('--dataset', default='NTU', type=str, help='indicate which dataset for pretrained')
parser.add_argument('--mode', default='camera', type=str, help='video or camera real time demo')
parser.add_argument('--show', default=True, type=bool, help='indicate if show image or not')
parser.add_argument('--num_classes', default=60, type=int, help='num of classes')

"---------------------  Video Demo Options   ---------------------"
parser.add_argument('--video_path', default='data/example/test_example.avi', type=str, help='path to test video')
parser.add_argument('--save_video', default=True, type=bool, help='indicate saveing video or not')
parser.add_argument('--out_path', default='data/output', type=str, help='path to save video')
parser.add_argument('--one_shot', default=False, type=bool, help='one video recognize only one time')
parser.add_argument('--step', default=1, type=int, help='sample step for recognition')

"---------------------  Camera Demo Options   ---------------------"
parser.add_argument('--modality', default='DR', type=str, help='D means Detect, R means Recognition')
parser.add_argument('--detect_sample_step', default=10, type=int, help='detect sample step')
parser.add_argument('--recognize_sample_step', default=2, type=int, help='recognize sample step')

"---------------------    Detect Options   -------------------------"
parser.add_argument('--cfg', default='yolo/cfg/person.cfg', type=str, help='path to darknet cfg file')
parser.add_argument('--detect_size', default=416, type=int, help='input size of detect net')
parser.add_argument('--detect_weights_path', default='yolo/weights/coco_person.pt', help='path to yolo weigths')
parser.add_argument('--conf_thres', default=0.5, type=float, help='confidence threshold')
parser.add_argument('--nms_thres', default=0.45, type=float, help='nms threshold')
parser.add_argument('--class_path', default='yolo/data/person.names', type=str, help='path to person names')

"---------------------	  Recognition Options ------------------------"
parser.add_argument('--recognize_weights_path', default='recognition/weights/best', help='path to recognition model weights')
parser.add_argument('--recognize_size', default=448, type=int, help='input size of recognition net')
parser.add_argument('--temporal_sample_length', default=32, type=int, help='temporal sample for recognition')

opt = parser.parse_args()
