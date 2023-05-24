import os
import random
import time
from glob import glob as glob

import cv2
import torch
import torchvision
from torch import nn

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
from models.common import Conv

from tqdm import tqdm

if torch.cuda.device_count() > 0:
    print("RUNNING ON GPU")
    DEVICE = torch.device('cuda')
else:
    print("RUNNING ON CPU")
    DEVICE = torch.device('cpu')


def load_model(weight, map_location=None):
    model = torch.load(weight, map_location=map_location)  # load
    model = model['model'].float().fuse().eval()
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    return model


def detect(model, test_path, save_dir, save_img=False):
    img_size = 640
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    # Get names (labels) that the model predicts
    names = model.module.names if hasattr(model, 'module') else model.names
    # randomly assign colours to each label
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if DEVICE.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(DEVICE).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    imgs = glob(os.path.join(test_path, "*.jpg"))
    for img_file in tqdm(imgs):
        # the model takes in RGB
        im0 = cv2.imread(img_file)  # need one that is not modified. This gets used for saving later on as well

        img = cv2.imread(img_file)  # load img used for actual model
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # refer to line 188 pf dataset.py from yolov5 repo

        # the resizing technique used by this project
        img = letterbox(img, (img_size, img_size), stride=32)[0]  # using the default values

        # testing with shape 640x640 which is the shape used during training
        # img = cv2.resize(img, (img_size, img_size))

        # change (HxWxC) -> (CxHxW)
        img_tensor = torchvision.transforms.ToTensor()(img)  # this takes care of normalization as well
        img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32

        # add dimension for batch if there is not one
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        t1 = time_synchronized()

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]  # returns y, None

        # perform NMS on the prediction to get list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.1, classes=None, agnostic=False)

        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            save_name = img_file.split("/")[-1].split(".")[0] + f"_output_{i}.jpg"
            s = ''
            s += f'Shape: {img_tensor.shape[2:]}'  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round().detach()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    try:
                        print(int(c))
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    except IndexError:
                        print(n, int(c))

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:
                        # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            cv2.imwrite(os.path.join(save_dir, save_name), im0)

    print(f"Results saved to {save_dir}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    classes = ["Aluminium foil",
               "Battery",
               "Aluminium blister pack",
               "Carded blister pack",
               "Other plastic bottle",
               "Clear plastic bottle",
               "Glass bottle",
               "Plastic bottle cap",
               "Metal bottle cap",
               "Broken glass",
               "Food Can",
               "Aerosol",
               "Drink can",
               "Toilet tube",
               "Other carton",
               "Egg carton",
               "Drink carton",
               "Corrugated carton",
               "Meal carton",
               "Pizza box",
               "Paper cup",
               "Disposable plastic cup",
               "Foam cup",
               "Glass cup",
               "Other plastic cup",
               "Food waste",
               "Glass jar",
               "Plastic lid",
               "Metal lid",
               "Other plastic",
               "Magazine paper",
               "Tissues",
               "Wrapping paper",
               "Normal paper",
               "Paper bag",
               "Plastified paper bag",
               "Plastic film",
               "Six pack rings",
               "Garbage bag",
               "Other plastic wrapper",
               "Single-use carrier bag",
               "Polypropylene bag",
               "Crisp packet",
               "Spread tub",
               "Tupperware",
               "Disposable food container",
               "Foam food container",
               "Other plastic container",
               "Plastic glooves",
               "Plastic utensils",
               "Pop tab",
               "Rope & strings",
               "Scrap metal",
               "Shoe",
               "Squeezable tube",
               "Plastic straw",
               "Paper straw",
               "Styrofoam piece",
               "Unlabeled litter",
               "Cigarette"
               ]
    weight = 'best.pt'

    save_dir = "outputs"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model = load_model(weight, map_location='cpu')
    half = False
    if half:
        model.half()  # to FP16

    test_path = "test_imgs"
    detect(model=model, test_path=test_path, save_img=True, save_dir=save_dir)
