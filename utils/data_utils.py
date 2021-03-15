import json
from dotmap import DotMap
import os
from collections import defaultdict

import cv2
def parse_json(json_file, config):
    with open(json_file) as file:
        data = DotMap(json.load(file))

    filenames = []
    bboxes = defaultdict(list)
    labels = []


    images = data.images
    # might not be neccessary as i think its already sorted, but im just gonna leave the code here anyway
    # images = sorted(images, key=lambda x: x['id'])  # makes sure the data is sorted by the id
    for index, image in enumerate(images):
        # needs the index to be same as the image_id
        # the reason why i made it like this is because it's already ordered
        assert index == image['id'], f"Images not in order after {index}"
        filename = os.path.join(config.data.root_dir, image["file_name"])
        filenames.append(filename)

    annotations = data.annotations
    # annotations = sorted(annotations, key=lambda x:x['id'])

    for index, annotation in enumerate(annotations):
        img_id = annotation['image_id']
        bbox = annotation['bbox']

        label = annotation['category_id']
        bboxes[img_id].append((bbox, label))

    return filenames, bboxes, labels


def crop_images_and_get_labels_for_supervised(filename, bboxes_and_labels):
    # going through all the images
    cropped_images = []
    labels = []
    img_bgr = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # going through all the objects in the given image
    for bbox, label in bboxes_and_labels:
        x1, y1, x2, y2 = bbox
        labels.append(label)
        cropped_images.append(img_rgb[y1:y2, x1:x2])

    return cropped_images, labels
