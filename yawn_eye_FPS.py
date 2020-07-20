# import the necessary packages
from __future__ import print_function
from __future__ import division
from imutils.video import WebcamVideoStream
import imutils
from imutils import face_utils
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame
import pyttsx3

engine = pyttsx3.init()
engine.say("System initialising")
engine.runAndWait()

print("Starting Program...")
detector = dlib.get_frontal_face_detector()

print("Initializing pretrained predictor...")
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat_2"
predictor = dlib.shape_predictor(PREDICTOR_PATH)


print("Initializing class for threading")
#Threading camera frames for better fps
from threading import Thread
class WebcamVideoStream:
    def __init__(self, src=0):
	# initialize the video camera stream and read the first frame
	# from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
	# initialize the variable used to indicate if the thread should
	# be stopped
        self.stopped = False

    def start(self):
	# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


'''
frame = vs.read()
frame = imutils.resize(frame, width=400)
cv2.imshow('Sample', frame )
'''
    
print("Initializing functions...")
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    #image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image, lip_distance #image_with_landmarks, lip_distance

def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("z.ogg")
    pygame.mixer.music.play()

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
######
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
   
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return ear

'''
cap = cv2.VideoCapture(0)
if(cap is None):
    print("Camera not detected!")
    exit()
'''

yawns = 0
yawn_status = False


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0
alarm=False

print("Starting Livestream....")
engine.say("Starting Livestream")
engine.runAndWait()

vs = WebcamVideoStream(src=0).start()
if(vs is None):
    print("Camera not detected!")
    exit()

print("Starting Loop...")

while True:
    #ret, frame = cap.read()
    #print("return value is"+str(ret))
    
    frame = vs.read()
    
    if (frame is None) :
        print(" Frame not received")
        continue
    
    #frame = imutils.resize(frame, width=400)
    frame_copy = frame.copy()
    frame_grey = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    #Starting Eye blink Detection
    dets = detector(frame_resized, 1)
    
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR+ rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            '''
            if cv2.waitKey(3)&0xFF==ord('l'):
                print(leftEyeHull)
                print('\n---> '+str(cv2.contourArea(leftEyeHull)))
            '''
            rightEyeHull = cv2.convexHull(rightEye)
            '''
            if cv2.waitKey(3) & 0xFF == ord('r'):
                print(rightEyeHull)
                print('\n---> '+str(cv2.contourArea(rightEyeHull)))
            '''
            #cv2.drawContours(frame_copy, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(frame_copy, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear>.25:
                #print (ear)
                total=0
                alarm=False
                cv2.putText(frame_copy, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                total+=1
                print("Eye close count -" + str(total))
                if total>3:
                    if not alarm:
                        alarm=True
                        
                        d=threading.Thread(target=start_sound)
                        d.setDaemon(True)
                        d.start()
                        
                        engine.say("Wake up. Please don't sleep")
                        engine.runAndWait()
                        print ("SLEEEEEPING!!!")
                        #cv2.putText(frame_copy, "drowsiness detect" ,(250, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                cv2.putText(frame_copy, "Eyes close".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame_copy, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
    cv2.imshow("image", frame_copy)


    # Starting Yawn Detection
    '''
    cv2.imshow('Yawn Detection', frame )
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
    '''
    
    image_landmarks, lip_distance = mouth_open(frame_resized)
    print("Lip distance = "+ str(lip_distance))
    if (lip_distance == 0) :
        print(" image_landmarks not received")
        continue
    
    #prev_yawn_status = yawn_status  
    
    if lip_distance > 15:
        
        yawn_status = True 
        #output_text = " Yawn Count: " + str(yawns + 1)
        output_text = " Yawn !"
        engine.say("Yawning detected. Please don't sleep")
        engine.runAndWait()
        print ("YAWNING!!!")

        cv2.putText(image_landmarks, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    #if prev_yawn_status == True and yawn_status == False:
        #yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
      
#cap.release()
cv2.destroyAllWindows()
vs.stop()

