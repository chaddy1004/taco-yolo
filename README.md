# taco-yolo
Writing YOLO object detection algorithm on [TACO dataset](http://tacodataset.org/) for trash detection

This repository (for now) is only used to hold only the minimum amount of code needed from [Yolov5 repository by Ultralytics](https://github.com/ultralytics/yolov5).

# Training
The training of model was done using aforementioned yolov5 repository. They offer a great [documentation on training with custom data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) that I used to follow.
However, there are some processing that must be done to follow the procedure.

## Data
TACO dataset unfortunately does not follow the format that yolov5 repo uses to train. Therefore, the re-formatting the data must be done.
[coco_to_yolo.py](https://github.com/chaddy1004/taco-yolo/blob/using_existing_repo/coco_to_yolo.py) does this.
Currently, it generates the yolov5 compatible text file for each image, containing the annotation and bounding box.
There is some additional manual work that needs to be done to completely separate the images with the label, which is listed as my TODO for future work.
For now, you can download the reformatted data for yolov5 [here]()

Once you have the processed data, you can pretty much follow their instruction word for word. The only subtle part that you need to watch out for is the parameters for training.
This pretty much takes care of the procedure until 3rd step (Organize Directories)

### Training Parameter
Their instruction assumes that you will use a pretrained yolov5 weight. This does not work for TACO as it has different number of annotations.
Therefore, although it says not recommended, you have to train from randomly initialized weight. The command for starting a training loop is as follows.
```bash
python train.py --img 640 --batch 4 --epochs 300 --data taco.yaml --weights '' --cfg yolov5s.yaml
``` 

Few notes:
- taco.yaml is available on this repo [here](https://github.com/chaddy1004/taco-yolo/blob/using_existing_repo/data/taco.yaml).
- I highly recommend keeping 640 as the image size as it seems to be the "default" parameter value
- cfg file can be swapped out depending on the yolo model that you choose (based on step 4)
- Obviously, batch and epoch parameter should be specified based on your available hardware specs.


# Predict
After training, you can find trained weights under runs > exp > weights.
Move the models to this repo, or use the one that came with this repo if you cannot/do not want to train on your own.

There is predict code on the orignal repo, but it was not what I wanted and had a lot of complicated code to encompass many situations.
Therefore, this repository only took the essential code that is needed for predicting a result given a single image.

The code under directories [models](https://github.com/chaddy1004/taco-yolo/tree/using_existing_repo/models) and [utils](https://github.com/chaddy1004/taco-yolo/tree/using_existing_repo/utils) are directly from the original repository, and are required in model loading and detection process.

detect.py is written based on the [code from the original repository](https://github.com/ultralytics/yolov5/blob/master/detect.py). It only has the functionality of predicting the result given a single image.


## Running on test images
The code performs prediction on the images under the directory, text_imgs. The predicted result (bounding boxes drawn on the original image)
gets saved under the directory outputs.
You can use the currently loaded sample images, or use your own images for testing.



