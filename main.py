# import libraries
import os
import random
import cv2
import pandas as pd
import numpy as np
# from RCNN import config # doesn't work! ask Shaun

from helperfunctions import calculate_IOU, visualiseSS, save_ROI, cowfaceBB, BBconversion


# Sets overall data directory path
DIR_PATH = "data" # should rename to source data

# Dictates the MAXIMUM number or region proposals allowed for training and inference (at the end)
REGION_PROPOSALS = 500 # test number use 2000 in final version
FINAL_PROPOSALS = 200

# Dictates how many images should be created out of each original image
# Want a positive bias per Girshick
MAX_COWFACE_YES = 10
MAX_COWFACE_NO = 5

# Input dimensions based on MobileNet v2 requirements
INPUT_DIMS = (224, 224)

# set selective segmentation
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Total counts for cow faces and counts for background images. Used for naming convention
countCowFaces = 0
countBackground = 0

# Where the new files will be saved
cowPath = 'results/CowFaces'
bkgPath = 'results/Background'

# Convert YOLO to X1,Y1,X2,Y2 and get the image
cowBBandImage = BBconversion(DIR_PATH)

for groundTruthBB, image in cowBBandImage:
    print(f'ground truth labels: {groundTruthBB}')
    # testBB = cowfaceBB(image, groundTruthBB) # prints image with bb

    # selective search over the image
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process()

    # proposed Regions list
    proposedRegions = []
    for x,y,w,h in boxes:
        # print(x,y,w,h)
        proposedRegions.append((x, y, x + w, y + h))

    # testSS = visualiseSS(image, boxes)

# make a counter that connects to cowface yes and no counts. Do ROI & IOU up to those numbers
    yesCowROI = 0
    noCowROI = 0

    for boxRegion in proposedRegions[:REGION_PROPOSALS]: 
        for BB in groundTruthBB:
            # (startX, startY, endX, endY) = BB
            iou = calculate_IOU(BB, boxRegion)
            
            print(f'IOU is {iou}')
            roi = None
            
            if iou > 0.1 and yesCowROI < MAX_COWFACE_YES:
                print(f"Face detected at IOU {iou}")
                (regionStartX, regionStartY, regionEndX, regionEndY) = boxRegion
                print(f"The ground truth is {BB}")
                print(f"The box region from selective search is {boxRegion}")
                roi = image[regionStartY:regionEndY, regionStartX:regionEndX]
                # print(f"Shape of region is {roi.shape()}")
                imageName = "cowfaceYes{}.png".format(yesCowROI)
                finalPath = os.path.sep.join([cowPath,imageName])
                # print(finalPath)
                countCowFaces += 1
                yesCowROI += 1

#            completeOverlap = regionStartX >= startX and regionStartY >= startY and regionEndX <= endX and regionEndY <= endY
#            if not completeOverlap and iou < 0.05 and noCowROI < MAX_COWFACE_NO:           
            elif iou < 0.05 and noCowROI < MAX_COWFACE_NO:
                # "Detected background"
                (regionStartX, regionStartY, regionEndX, regionEndY) = boxRegion
                roi = image[regionStartY:regionEndY, regionStartX:regionEndX]
                imageName = "cowfaceNo{}.png".format(noCowROI)
                finalPath = os.path.sep.join([bkgPath,imageName])
                countBackground += 1
                noCowROI += 1

            # else:
            #     print("Inbetweener")

            if roi is not None:
                # print("Region of interest found and will be saved.")
                roi = cv2.resize(roi, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(finalPath, roi)
                print(f'Saved file {finalPath}')

        if noCowROI == MAX_COWFACE_NO and yesCowROI == MAX_COWFACE_YES:
            break
    if noCowROI == MAX_COWFACE_NO and yesCowROI == MAX_COWFACE_YES:
        break
    break

# for e, regions in enumerate(boxes):
#     # loop over region proposal count
#     if e < REGION_PROPOSALS:
#         (regionStartX, regionStartY, regionEndX, regionEndY) = regions
#         # loop over each ground truth bb
#         for groundTruthBox in groundTruthBB:
#             # calculate IOU
#             iou = calculate_IOU(groundTruthBox, regions)
#             # (startX, startY, endX, endY) = groundTruthBox # gets values we need
#             imgROI = None
#             # 1. if we haven't collected all our pos samples and the iou is > 70% grab that sample as a pos
#             if iou < 0.7 and yesCowROI < MAX_COWFACE_YES:
#                 imgROI = image[regionStartY:regionEndY, regionStartX:regionEndX]
#                 if imgROI is not None:
#                     try:
#                         save_ROI(imgROI, countCowFaces, cowPath, Face = True) 
#                         countCowFaces += 1
#                         yesCowROI += 1
#                     except Exception as e:
#                         print(e)
#                         print("Exception raised when processing yes cow ROI")
#                         continue
#             # 2. we need to check for cases of total overlap before getting the neg samples
#             # completeOverlap = regionStartX >= startX and regionStartY >= startY and regionEndX <= endX and regionEndY <= endY

#             # 3. if iou is , 0.05 and is not complete overlap, grab that sample as a neg
#             # removed and not completeOverlap
#             if iou < 0.05 and not completeOverlap and noCowROI < MAX_COWFACE_NO:
#                 imgROI = image[regionStartY:regionEndY, regionStartX:regionEndX]
#                 if imgROI is not NONE: 
#                     try:
#                         save_ROI(imgROI, countBackground, bkgPath, Face = False)
#                         countBackground += 1
#                         noCowROI += 1
#                     except Exception as e:
#                         print(e)
#                         print("Exception raised when processing no cow ROI")
#                         continue                

print(f'Yes Cow ROI count is {yesCowROI}')
print(f'No Cow ROI count is {noCowROI}')
print(f'Total background images {countBackground}')
print(f'Total cow face images {countCowFaces}')