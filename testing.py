import numpy as np
import cv2
import uinput
import time
import matplotlib.pyplot as plt
import random
import System
import Trainer

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
OFFSET_X = 0
OFFSET_Y = 50

INITIA_SLEEP=3

MASK_IMG = 'mask.png'



def runTrainingRecorder():
    tr = Trainer.TrainingRecorder(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,1.0/20.0,"outputDir")

    # tr.loadRawData()
    # tr.balanceData()
    # return
    tr.start()

    last_update = time.time()
    last_time = time.time()
    fps = 0
    while(True):
        fps = 0.9*fps+0.1*1.0/(time.time()-last_time)
        last_time = time.time()

        tr.process()

        if time.time()-last_update > 2:
            last_update = time.time()
            print "FPS: {}".format(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    tr.save()
    tr.loadRawData()
    tr.balanceData()
    return


def runControl():
    s = System.System(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,MASK_IMG)
    s.start()
    last_update = time.time()
    last_time = time.time()
    fps = 0
    while(True):
        fps = 0.9*fps+0.1*1.0/(time.time()-last_time)
        last_time = time.time()

        s.process()

        if time.time()-last_update > 2:
            last_update = time.time()
            print "FPS: {}".format(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    s.stop()

for i in range(INITIA_SLEEP):
    print INITIA_SLEEP-i
    time.sleep(1)

runTrainingRecorder()
#runControl()
