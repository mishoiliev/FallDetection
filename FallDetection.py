import cv2
import numpy as np
import time
import argparse
import json

def Parser():
    parser = argparse.ArgumentParser(description="Set a method for Fall Detection")
    parser.add_argument("--mode", choices= ['multi', "single"],required=True, type = str, help = "Mode of detection")
    parser.add_argument("--height", default = 60, type = int, 
                        help = "Distance between the torso and the head in cm. \nType '--height=<number>'")

    args = parser.parse_args()

    mode = args.mode
    person_Height = args.height
    return mode,person_Height

MODE = "COCO"

if MODE is "COCO":
    protoFile = "models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.01
FallDetected = False

input_source = "models/sample_video1.mp4"
cap = cv2.VideoCapture(input_source)
fps = cap.get(cv2.CAP_PROP_FPS)
hasFrame, frame = cap.read()

#vid_writer = cv2.VideoWriter('models/output4.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

headHeight = 0
torsoHeight = 0
torso_list = []
head_list = []


def DrawPoints(count = 0):
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            #cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)


    #if points.append(None) -> TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
    if None not in points:
        headHeight = np.array(points[0][1])
        torsoHeight = np.array(points[8][1])
    #draws circle and shows y value
        cv2.circle(frame, points[0], 8 , (0,255,0), thickness=-1, lineType = cv2.FILLED)
        cv2.circle(frame, points[8], 8 , (0,255,0), thickness=-1, lineType = cv2.FILLED)
        cv2.line(frame, points[0], points[8], (0,255,255), 3, lineType=cv2.LINE_AA)
        cv2.putText(frame, "{}".format(headHeight), points[0], cv2.FONT_HERSHEY_COMPLEX, .6, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, "{}".format(torsoHeight), points[8], cv2.FONT_HERSHEY_COMPLEX, .6, (255, 50, 0), 2, lineType=cv2.LINE_AA)

        #get int value for y
        return int(headHeight), int(torsoHeight)
    else:
        return int(headHeight_prev), int(torsoHeight_prev)

    # Draw Skeleton
#    for pair in POSE_PAIRS:
#         partA = pair[0]
#         partB = pair[1]

#         if points[partA] and points[partB]:
#             cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
#             cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
#             cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

def DetectFall(headHeight, headHeight_prev, fps, torsoHeight, pixel_length, FallDetected):
        #point y at curr frame - point y at previous
    delta_distance = torsoHeight - torsoHeight_prev
    change_per_sec = delta_distance*fps

    if mode == "multi":
        if change_per_sec > inHeight/2:
            if headHeight > inHeight/3:
                if torsoHeight - headHeight < 70:
                    FallDetected = True
                    return True
                else:
                    return False
        else:
            return False
    elif mode == "single":
        speed_per_sec = change_per_sec*pixel_length
        if speed_per_sec > 200:
            if torsoHeight - headHeight < 70:
                FallDetected = True
                return True
            else:
                return False
        else:
            return False

def DetectGetUp():
    delta_distance = torsoHeight - headHeight
    if delta_distance > 70:
        return True
    else:
        return False

if __name__ == '__main__':
    while cv2.waitKey(1) < 0:
        t = time.time()
        mode, person_Height = Parser()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            cv2.destroyAllWindows()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        headHeight_prev = headHeight
        torsoHeight_prev = torsoHeight
        headHeight, torsoHeight = DrawPoints()
        head_list.append(headHeight)
        torso_list.append(torsoHeight)
        try:
            pixel_length = float(person_Height/(torsoHeight - headHeight))
        except:
            pass
            #pixel_length = float(person_Height/(torsoHeight_prev - headHeight_prev))
        #error check
        print("Head Height: {}\nHead Heigh previous: {} \nTorso Height: {}".format(headHeight,headHeight_prev, torsoHeight))

        if DetectFall(headHeight, headHeight_prev, fps, torsoHeight, pixel_length, FallDetected):
            cv2.putText(frame, "Fall Detected", (50,50), 
                    cv2.FONT_HERSHEY_COMPLEX, .8, (0,0,255), 2, lineType=cv2.LINE_AA)

        if FallDetected:
            if DetectGetUp():
                cv2.putText(frame, "Get Up Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                .8, (0,255,0), 2, lineType=cv2.LINE_AA)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), 
                    (30, 20), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        #cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)

        #vid_writer.write(frame)

    heights = {
        "Video" : input_source,
        "Fall Detected" : FallDetected,
        "Head Height" : head_list,
        "Torso Height" : torso_list
    }


    with open('models/FallDetection/HeightPoints.txt', 'w') as json_file:
        json.dump(heights, json_file)
    #vid_writer.release()