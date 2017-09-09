import numpy as np
import cv2
import uinput
import time
import matplotlib.pyplot as plt
import random
import System
import Trainer
import tensorflow as tf
from alexNet import alexnet
import argparse

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
OFFSET_X = 0
OFFSET_Y = 50
PROCESS_WIDTH = 80
PROCESS_HEIGHT = 60

INITIA_SLEEP=3

MASK_IMG = 'mask.png'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

ID = 1
EPOCHS = 10
MODEL_DIR = 'models'
MODEL_NAME = 'model_alexnet'
MODEL = alexnet(PROCESS_WIDTH,PROCESS_HEIGHT,0.001,'{}/{}'.format(MODEL_DIR,MODEL_NAME))
RUN_NAME = "{}-epocs-{}-id-{}".format(MODEL_NAME,EPOCHS,ID)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--recordTraining',action='store_true')
    parser.add_argument('--play',action='store_true')
    args = parser.parse_args()
    print args

    for i in range(INITIA_SLEEP):
        print INITIA_SLEEP-i
        time.sleep(1)

    if args.train:
        runTraining()
    elif args.recordTraining:
        runTrainingRecorder()
    elif args.play:
        runControl()

def runTraining():
    tr = Trainer.TrainingRecorder(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT,1.0/20.0,"trainData")
    tr.trainModel(MODEL,MODEL_DIR,MODEL_NAME,RUN_NAME,EPOCHS)

def runTrainingRecorder():
    tr = Trainer.TrainingRecorder(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT,1.0/20.0,"trainData")

    # tr.loadRawData()
    # tr.balanceData()
    #tr.trainModel()
    #return
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

    MODEL.load('{}/{}.tflearn'.format(MODEL_DIR,MODEL_NAME))


    s = System.System(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT,MASK_IMG,MODEL)
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


main()
