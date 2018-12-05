from net.i3d_encoder import I3D_with_encoder, split_I3D
from net.I3D import I3D, I3D_convlstm
from net.ResNet3D import resnet101,resnet50,resnet152
from ResNet_encoder import resnet101_, resnet50_, resnet152_



'''
def load_model(args):
	if args.method == 'global':
		if args.basicnet == 'inception':
			model = I3D(num_classes=args.num_classes, input_size=args.input_size, modality=args.modality, dropout_prob=args.dropout_prob)
		elif args.basicnet == 'resnet101':
			model = resnet101(sample_size=args.input_size, sample_duration=args.time_sample, num_classes=args.num_classes)
		elif args.basicnet == 'resnet50':
			model = resnet50(sample_size=args.input_size, sample_duration=args.time_sample, num_classes=args.num_classes)
		elif args.basicnet == 'resnet152':
			model = resnet152(sample_size=args.input_size, sample_duration=args.time_sample, num_classes=args.num_classes)
		else:
			raise ValueError('Got a wrong value for basicnet')
	return model
'''

def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('conv3d') != -1:
        model.weight.data.xavier_normal()
    elif classname.find('batch3d') != -1:
        model.weight.data.xavier_normal()
        model.bias.data.fill_(0)

def show_model(model):
    model_dict = model.state_dict()
    for name, ele in model_dict.items():
        #print(name)	
        print(ele.requires_grad)


def fix_layer(model):
    print('model fixing')
    model_dict = model.state_dict()
    for name, value in model_dict.items():
        if str(name).split('.')[0] != 'convlstm' and str(name).split('.')[0] != 'conv3d_0c_1x1_':
            #print(name)
            value.requires_grad = False
