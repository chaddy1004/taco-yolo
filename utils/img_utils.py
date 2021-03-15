import cv2

def load_image(filename, bboxes_and_labels):
    # going through all the images
    cropped_images = []
    labels = []
    img_bgr = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # going through all the objects in the given image
    for bbox, label in bboxes_and_labels:
        x1, y1, x2, y2 = 
        labels.append(label)
        cropped_images.append(img_rgb[])





    bboxes[img_id].append((bbox, label))