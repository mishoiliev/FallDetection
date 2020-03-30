import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from FallDetection import fps
from FallDetection import inHeight
from FallDetection import FallDetected

head_height_prev = 0
torso_height_prev = 0
personHeight = 60

def DetectFallMulti(headHeight, torsoHeight, torsoHeightPrev):
    delta_distance = torsoHeight - torsoHeightPrev
    change_per_sec = delta_distance*fps

    if change_per_sec > inHeight/2:
        if headHeight > inHeight/3:
            if torsoHeight - headHeight < 70:
                FallDetected = True
                return True
            else:
                return False

def DetectFallSingle(headHeight, torsoHeight, torsoHeightPrev):
    delta_distance = torsoHeight - torsoHeightPrev
    change_per_sec = delta_distance*fps

    speed_per_sec = change_per_sec*pixel_length
    if speed_per_sec > 200:
        if torsoHeight - headHeight < 70:
            FallDetected = True
            return True
        else:
            return False

with open('models/FallDetection/HeightPoints.txt', 'r') as json_file:
    heights = json.load(json_file)
head_heights = heights['Head Height']
torso_heights = heights['Torso Height']
for i in range(len(head_heights)):
    if i>0:
        print(head_heights[i], torso_heights[i])
        head_height = head_heights[i]
        torso_height = torso_heights[i]

        head_height_prev = head_heights[i-1]
        torso_height_prev = torso_heights[i-1]
        
        try:
            pixel_length = float(personHeight/(torso_height - head_height))
        except:
            pass

        if DetectFallMulti(head_height, torso_height, torso_height_prev) == True:
            detect = {"Fall Detected": FallDetected}
            with open('models/FallDetection/FallDetection.txt', 'w') as json_file:
                json.dump(detect, json_file)

