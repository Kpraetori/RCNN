import numpy as np
import cv2
import os
from pathlib import Path
from typing import List, Tuple

class BBconversion:
    """Dataloader currently the directory path and augmentation boleen. 
    Images are pulled and annotations are taken from the text file and converted from YOLO to xyxy. Return is a tuple with a list and np array. This makes the daa ready for Selective Segmentation."""
    def __init__(self, directory: str):
        self.directory = directory
        self.annotation_path = list(Path(self.directory).glob('*.txt')) # path for annotations

    @staticmethod # used for functions with no class parameter like helper and maths functions
    def _format_annotations(annotation_path: str): # _ first because function kept within
        with open(annotation_path, 'r') as ann_file:
            annotation_strings = [line for line in ann_file] # for each line in the txt file, make a string
        # Now take the strings, pull each value, and add them to a list
        formatted_annotations = []
        for ann_strings in annotation_strings:
            c, x, y, w, h = ann_strings.split(' ') 
            c = int(c)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            formatted_annotations.append([c, x, y, w, h])
            # print(f'The YOLO format is: {formatted_annotations}')
        return formatted_annotations

    def _open_image(self, annotation_path: str) -> np.array:
        path_no_ext = os.path.splitext(annotation_path)[0] # split the annotation path and select not the extension
        file_name = os.path.basename(path_no_ext)
        # print(file_name)
        dir_path = Path(self.directory).glob(f'{file_name}*')
        image_path = [d_path for d_path in dir_path if not d_path.name.endswith('.txt')]
        if len(image_path) > 0:
            image_path = image_path[0]
        im = cv2.imread(str(image_path))
        # cv2.imshow('Printed Image', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return im

    @staticmethod
    def convert_yolo_to_xyxy(anns: List, img: np.array):
        xyxy_anns = []
        dh, dw, _ = img.shape
        for bb in anns:
            _, x, y, w, h = bb    
            x1 = int((x - w / 2) * dw)
            x2 = int((x + w / 2) * dw)
            y1 = int((y - h / 2) * dh)
            y2 = int((y + h / 2) * dh)

            # x1 = (x * dw) - (w * dw) / 2
            # x2 = (x * dw) + (w * dw) / 2
            # y1 = (y * dh) - (h * dh) / 2
            # y2 = (y * dh) + (h * dh) / 2
            xyxy_anns.append([x1, y1, x2, y2])
        # print(f'The xy coordinates are: {xyxy_anns}')        
        return xyxy_anns

    def __len__(self): # having a len fun is required. 
        return len(self.annotation_path)

    # idx is for index. the -> shows what format I think...
    def __getitem__(self, idx: int) -> Tuple[List, np.array]: 
        yoloann = self._format_annotations(self.annotation_path[idx])
        img = self._open_image(self.annotation_path[idx])
        ann = self.convert_yolo_to_xyxy(yoloann, img)
        return ann, img




# ra = [3., 3., 5., 5.]
# rb = [1., 1., 4., 3.5]
# intersection of these should be 0.5


### PYIMAGE SEARCH IOU - used to debug code when saving wasn't working
def compute_iou(boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the intersection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


### MY IOU
def calculate_IOU(coordinatesA, coordinatesB):
    """Calculate intersection over union with inputs of ground truth and predicted bb provided in any order. Assumes both coordinates inputs are in xyxy list format."""
    # Use largest and smallest (max and min) to get intersection coord
    greaterX = max(coordinatesA[0], coordinatesB[0])
    lesserX = min(coordinatesA[2], coordinatesB[2])
    greaterY = max(coordinatesA[1], coordinatesB[1])
    lesserY = min(coordinatesA[3], coordinatesB[3])
   
    # area of the rectangle of intersection
    intersection = (lesserX - greaterX) * (lesserY - greaterY)
    # print(f'Intersection is {intersection}')

    # Area of both boxes and get union
    areaA = (coordinatesA[2] - coordinatesA[0]) * (coordinatesA[3] - coordinatesA[1])
    # print(f'Area of A is {areaA}')
    areaB = (coordinatesB[2] - coordinatesB[0]) * (coordinatesB[3] - coordinatesB[1])
    # print(f'Area of B is {areaB}')
    union = float(areaA + areaB - intersection)
    # print(f'Union is {union}')

    iou = intersection/union
    # print(f'IOU is {iou}')
    return iou

# reply = calculate_IOU(ra,rb)
# print(reply)

# this prints the ss results on the cow image to see that it works
def visualiseSS(yourImage, boxesVariable):
    for region in range(0, len(boxesVariable), 30):
        copy = yourImage.copy()
        for (x,y,w,h) in boxesVariable[region: region + 30]:
            cv2.rectangle(copy, (x,y), (x + w, y + h), (0,255,100), 2)
        cv2.imshow("Sample Regions on Image", copy)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
    	    break

# this prints the bb on the cow image to see if it works
def cowfaceBB(img, ann):
        for region in range(0, len(ann), 30):
            copy = img.copy()
        for (x1,y1,x2,y2) in ann[region: region + 30]:
            cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,100), 2)
        cv2.imshow("BB Regions on Image", copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# For testing new function
# CounterA = 2
# CounterB = 4
# Data = cv2.imread('data/1.mp4_frame_2790.jpg')


INPUT_DIMS = (224, 224)
cowFacePath = 'results/Cowfaces'
backgroundPath = 'results/Background'

# code requires a face count and background count variable
def save_ROI(RegionData, counterName, savePath: str, Face: bool):
    if Face is True:
        answer = 'Yes'
    else:
        answer = 'No'
    imageName = "cowface{}{}.png".format(answer, counterName)
    finalPath = os.path.sep.join([savePath,imageName])
    resize_image = cv2.resize(RegionData, (224,224), interpolation=cv2.INTER_CUBIC)
    return cv2.imwrite(finalPath, resize_image) and print(f'{imageName} is saved')


# testA = save_ROI(Data, CounterA, cowFacePath, Face=True)
# testB = save_ROI(Data, CounterB, backgroundPath, Face=False)


# Should NMS be a function or a class? Need to do tutoorials on NMS...
class NMS:
    pass