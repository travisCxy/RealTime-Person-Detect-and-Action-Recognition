import argparse
import time

from models import *
from yolo_utils.datasets import *
from yolo_utils.utils import *
from detect_opt import opt

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')



def detect(opt, image_file, label_file):
    os.system('rm -rf ' + opt.output_folder)
    os.makedirs(opt.output_folder, exist_ok=True)

    # Load model
    model = Darknet(opt.cfg, opt.img_size)

    weights_path = 'weights/yolov3_person.pt'
    if weights_path.endswith('.weights'):  # saved in darknet format
        load_weights(model, weights_path)
    else:  # endswith('.pt'), saved in pytorch format
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        del checkpoint

        # current = model.state_dict()
        # saved = checkpoint['model']
        # # 1. filter out unnecessary keys
        # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
        # # 2. overwrite entries in the existing state dict
        # current.update(saved)
        # # 3. load the new state dict
        # model.load_state_dict(current)
        # model.to(device).eval()
        # del checkpoint, current, saved

    model.to(device).eval()

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    #dataloader = load_images(opt.image_folder, batch_size=opt.detect_batch_size, img_size=opt.img_size)
    dataloader = load_images(image_file, batch_size=opt.detect_batch_size, img_size=opt.img_size)
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    for batch_i, (img_paths, img) in enumerate(dataloader):
        print(batch_i, img.shape, end=' ')

        # Get detections
        with torch.no_grad():
            chip = torch.from_numpy(img).unsqueeze(0).to(device)
            pred = model(chip) # pred is of shape [1, 10647, 6] 6 means six attributes of box(x, y, w, h, p, c)
            pred = pred[pred[:, :, 4] > opt.conf_thres]   #通过类别置信度过滤掉一部分box

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), opt.conf_thres, opt.nms_thres)#通过nms算法过滤掉一部分box
                img_detections.extend(detections)
                imgs.extend(img_paths)

        print('Batch %d... (Done %.3f s)' % (batch_i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if len(img_detections) == 0:
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        if opt.plot_flag:
            img = cv2.imread(path)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x
        # Draw bounding boxes and labels of detections
        if detections is not None:
            #print(detections)
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
            results_txt_path = results_img_path + '.txt'
            if os.path.isfile(results_txt_path):
                os.remove(results_txt_path)

            for i in unique_classes:
                n = (detections[:, -1].cpu() == i).sum()
                print('%g %ss' % (n, classes[int(i)]))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
               

                # write to file
                if opt.txt_out:
                    label_file.write(('%s %g %g %g %g %g %g \n') % (path, x1, y1, x2, y2, cls_pred, cls_conf * conf))

                if opt.plot_flag:
                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(cls_pred)], conf)
                    color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                    plot_one_box([x1, y1, x2, y2], img, label=label, color=color)

        if opt.plot_flag:
            # Save generated image with detections
            cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    video_dir = 'ntu'
    video_file_list = os.listdir(video_dir)
    video_file_list.sort()
    label_dir = 'ntu_label'
    for image_file_name in video_file_list:
        label_file = open(label_dir+'/'+image_file_name+'.txt', 'a')
        image_file = os.path.join(video_dir,image_file_name)
        detect(opt, image_file, label_file)
