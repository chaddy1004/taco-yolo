import os
import json
from dotmap import DotMap
from collections import defaultdict
from glob import glob
import shutil
import cv2


def move_data(source_dir, target_dir, data):
    img_id_dict = defaultdict()
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for image in data.images:
        id = image.id
        new_img_file = os.path.join(target_dir, f"{id}.jpg")

        txt_file = os.path.join(target_dir, f"{id}.txt")

        curr_file = os.path.join(source_dir, image.file_name)
        shutil.copy(curr_file, new_img_file)

        img_id_dict[id] = (image.height, image.width)

        f = open(txt_file, "w+")
        f.close()

        print(curr_file, new_img_file)
    return img_id_dict


def parse_data(target_dir, data, img_id_dict):
    """
    # One row per object
    # Each ros is class x_center y_center width height format
    # box coordinates must be in normaized xywh format (from 0 to 1). 
    # Class numebrs are zero-indexed (start from 0)
    """
    annotations = data.annotations
    for annotation in annotations:
        img_id = annotation.image_id
        img_h, img_w = img_id_dict[img_id]
        x_start, y_start, w, h = annotation.bbox

        x_center = int(x_start + w // 2)
        y_center = int(y_start + h // 2)

        x_norm = x_start / img_w
        w_norm = w / img_w
        y_norm = y_start / img_h
        h_norm = h / img_h
        img_file = os.path.join(target_dir, f"{img_id}_rect.JPG")
        if not os.path.exists(img_file):
            img_file = os.path.join(target_dir, f"{img_id}.jpg")

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_point = (int(x_start), int(y_start))
        end_point = (int(x_start + w), int(y_start + h))
        colour = (255, 0, 0)
        thickness = 3

        img = cv2.rectangle(img=img, pt1=start_point, pt2=end_point, color=colour, thickness=thickness)
        img = cv2.circle(img=img, center=(x_center, y_center), radius=3, color=colour, thickness=5)

        cv2.imwrite(os.path.join(target_dir, f"{img_id}_rect.JPG"), img)

        label = annotation.category_id
        id_txt_file = os.path.join(target_dir, f"{img_id}.txt")
        with open(id_txt_file, "a") as f:
            line = f"{label} {x_norm} {y_norm} {w_norm} {h_norm}\n"
            f.write(line)

    return


if __name__ == '__main__':
    source_json = "annotations/annotations_original.json"

    source_dir = "/Users/chaddy/dev/TACO/data/"
    target_dir = "/Users/chaddy/dev/TACO/data_new/"

    with open(source_json) as file:
        data = DotMap(json.load(file))

    img_id_dict = move_data(source_dir=source_dir, target_dir=target_dir, data=data)
    parse_data(target_dir=target_dir, data=data, img_id_dict=img_id_dict)
