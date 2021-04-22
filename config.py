
##### Ask Shaun. How do I get this to connect with main.py? It gives me an error when I try to import
import os

##### Setup Data #####
# Sets overall data directory path
DIR_PATH = "data"

# Create a path for the segmented images to go
SEGMENTED_PATH = "dataset"
COWFACE_YES = os.path.join(SEGMENTED_PATH, "cowface_yes")
COWFACE_NO = os.path.join(SEGMENTED_PATH, "cowface_no")

# Dictates the MAXIMUM number or region proposals allowed for training and inference (at the end)
REGION_PROPOSALS = 2000
FINAL_PROPOSALS = 200

# Dictates how many images should be created out of each original image
# Want a positive bias per Girshick
MAX_COWFACE_YES = 30
MAX_COWFACE_NO = 10

##### Model data will go below #####

# Network input dimensions required by MobileNet
# Path for models stuff
# What else should go here?