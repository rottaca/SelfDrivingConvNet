import numpy as np
import cv2
import uinput
import time
import matplotlib.pyplot as plt
import random
import System
import Trainer
import tensorflow as tf
import keyboard
import argparse
import sys

from alexNet import alexnet
from myNet import mynet

REC_SCREEN_WIDTH = 1024
REC_SCREEN_HEIGHT = 768
REC_OFFSET_X = 0
REC_OFFSET_Y = 50

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
OFFSET_X = 0
OFFSET_Y = 50

PROCESS_WIDTH = 135
PROCESS_HEIGHT = 100

INITIAL_SLEEP=3

# Keys to be looked for
KEYS=[
'up',
'down',
'left',
'right'
]
# Mapping from steering commands (index) to valid key combinations
# e.g. multiple commands to turn right or left are possible
KEYS_TO_ONEHOT=[
              [[1,0,0,0]],                      # driving forwards
             # [[0,1,0,0]],                     # driving backwards
              [[1,0,1,0],[0,0,1,0]],            # driving left
              [[1,0,0,1],[0,0,0,1]]             # driving right
              ]
# Names for one hot vectors
# -> Steering command names
ONEHOT_NAMES = [
                "fwd",
            #   "bwd",
                "left",
                "right"
                ]

MASK_IMG = 'mask.png'

config=tf.ConfigProto()
config.log_device_placement=True
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

ID = "trackmania"
EPOCHS = 7
TENSORBOARD_DIR = "tensorboard/"
MODEL_DIR = 'models'
MODEL_NAME = 'model_mynet'
MODEL = mynet(PROCESS_WIDTH,PROCESS_HEIGHT,len(KEYS_TO_ONEHOT),0.001,'{}/{}'.format(MODEL_DIR,MODEL_NAME),TENSORBOARD_DIR)
RUN_NAME = "{}-epocs-{}-id-{}".format(MODEL_NAME,EPOCHS,ID)
CONTINUE_FROM_CHECKPOINT=True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--play',action='store_true')
    parser.add_argument('run_suffix', nargs='?')
    args = parser.parse_args()
    print args

    for i in range(INITIAL_SLEEP):
        print INITIAL_SLEEP-i
        time.sleep(1)

    if args.train:
        if args.run_suffix is None:
            print "Run name missing"
            exit(1)
        runTraining(args.run_suffix)
    elif args.record:
        runTrainingRecorder()
    elif args.play:
        runControl()

def runTraining(run_suffix):
    tr = Trainer.TrainingRecorder("trainData",KEYS,KEYS_TO_ONEHOT)
    tr.setupDimensions(REC_SCREEN_WIDTH,REC_SCREEN_HEIGHT,REC_OFFSET_X,REC_OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT)
    tr.trainModel(MODEL,MODEL_DIR,MODEL_NAME,"{}_{}".format(RUN_NAME,run_suffix),EPOCHS,MASK_IMG,CONTINUE_FROM_CHECKPOINT)

def runTrainingRecorder():
    tr = Trainer.TrainingRecorder("trainData",KEYS,KEYS_TO_ONEHOT)
    tr.setupDimensions(REC_SCREEN_WIDTH,REC_SCREEN_HEIGHT,REC_OFFSET_X,REC_OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT)
    tr.start(1.0/30.0,)

    last_update = time.time()
    last_time = time.time()
    fps = 0
    while(True):
        if tr.process():
            fps = 0.9*fps+0.1*1.0/(time.time()-last_time)
            last_time = time.time()

        if time.time()-last_update > 1:
            last_update = time.time()
            sys.stdout.write("\rFrame count: {}, FPS: {}".format(len(tr.trainData),fps))
            sys.stdout.flush()

        if keyboard.is_pressed('q'):
            cv2.destroyAllWindows()
            break
    tr.save()
    tr.loadRawData()
    tr.balanceData()
    return


def runControl():

    MODEL.load('{}/{}.tflearn'.format(MODEL_DIR,MODEL_NAME))

    s = System.System(SCREEN_WIDTH,SCREEN_HEIGHT,OFFSET_X,OFFSET_Y,PROCESS_WIDTH,PROCESS_HEIGHT,MASK_IMG,MODEL,KEYS,KEYS_TO_ONEHOT,ONEHOT_NAMES)
    s.start()
    last_update = time.time()
    last_time = time.time()
    fps = 0
    while(True):

        if s.process():
            fps = 0.9*fps+0.1*1.0/(time.time()-last_time)
            last_time = time.time()
            print "FPS: {}".format(fps)

        if keyboard.is_pressed('q'):
            cv2.destroyAllWindows()
            break
    s.stop()


main()
