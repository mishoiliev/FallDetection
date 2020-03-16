import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from FallDetection import DetectFall
from FallDetection import *

head_height_prev = 0
torso_height_prev = 0

with open('models/FallDetection/HeightPoints.txt', 'r') as json_file:
    heights = json.load(json_file)
head_heights = heights['Head Height']
torso_heights = heights['Torso Height']
for i in range(len(head_heights)):
    print(head_heights[i], torso_heights[i])
    head_height = head_heights[i]
    torso_height = torso_heights[i]
    try:
        head_height_prev = head_heights[i-1]
        torso_height_prev = torso_heights[i-1]
    except:
        pass

if heights["Fall Detected"] == True:
    print("Fall Detected")