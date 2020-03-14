import numpy as np
import cv2
import json
import matplotlib

with open('models/FallDetection/HeightPoints.txt', 'r') as json_file:
    heights = json.load(json_file)
    head_heights = heights['Head Height']
    torso_heights = heights['Torso Height']
    for i, j in zip(head_heights,torso_heights):
        print(i, j)