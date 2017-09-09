import numpy as np
import cv2
import uinput
import time
import random
import keyboard
import os

import tflearn

import MyUtils

class TrainingRecorder:

    def __init__(self, screen_w, screen_h, screen_x, screen_y, proc_width, proc_height, interval, outputDir):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.proc_width= proc_width
        self.proc_height= proc_height
        self.scc = MyUtils.ScreenCapture({'width': screen_w, 'top': screen_y, 'height': screen_h, 'left': screen_x})
        self.interval = interval
        self.outputDir = outputDir
        self.rawDataFileName = outputDir + "/rawData.npy"
        self.balancedDataFileName = outputDir + "/balancedData.npy"

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    def start(self):
        self.last_time = time.time()
        self.trainData = []
        self.loadRawData()

    def save(self):
        print("Saving data of size: {}".format(len(self.trainData)))
        np.save(self.rawDataFileName,self.trainData)

    def process(self):
        if time.time()-self.last_time > self.interval:
            self.last_time = time.time()
            img = self.scc.grab()
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(self.proc_width,self.proc_height))

            if keyboard.is_pressed('up'):
                if keyboard.is_pressed('left'):
                    keys = [1,0,0]
                elif keyboard.is_pressed('right'):
                    keys = [0,0,1]
                else:
                    keys = [0,1,0]
            elif keyboard.is_pressed('down'):
                #keys = [0,0,0,1]
                return
            else:
                return

            self.trainData.append([img,keys])
            print "Frame count: " + str(len(self.trainData))

    def loadRawData(self):
        if os.path.exists(self.rawDataFileName):
            self.trainData = list(np.load(self.rawDataFileName))
            print "Loaded data with {} entries".format(len(self.trainData))
        else:
            print "No prerecorded data to load."

    def balanceData(self):
        keyCounts = np.array([0,0,0])
        random.shuffle(self.trainData)
        fwd = []
        left =  []
        right = []
        print "Extracting data..."
        for idx, data in enumerate(self.trainData):
            img = data[0]
            keys = data[1]
            #cv2.imshow("img", img)
            #cv2.resizeWindow('img', self.proc_height0,self.proc_height0)

            keyCounts += np.array(keys)
            if keys == [1,0,0]:
                left.append([img,keys])
            elif keys == [0,1,0]:
                fwd.append([img,keys])
            elif keys == [0,0,1]:
                right.append([img,keys])
            #print "\r {}/{} ({}%)".format(idx,len(self.trainData),100.0*idx/len(self.trainData))

        print "Done."
        print "Key counts: " + str(keyCounts)
        minCnt = min(keyCounts)
        print "Minimum samples per class: " + str(minCnt)
        print "Balancing..."
        fwd = fwd[:minCnt]
        left = left[:minCnt]
        right = right[:minCnt]

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        print "fwd"
        time.sleep(3)
        for data in fwd:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(1)

        print "left"
        time.sleep(3)
        for data in left:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(1)

        print "right"
        time.sleep(3)
        for data in right:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(1)


        balancedData = fwd + left + right
        print("Saving balanced of size: {}".format(len(balancedData)))
        random.shuffle(balancedData)
        np.save(self.balancedDataFileName,balancedData)
        print "Done."

    def trainModel(self, model, model_dir, model_name, run_id, n_epoch):

        balancedData = list(np.load(self.balancedDataFileName))
        print "Loaded data with {} entries".format(len(balancedData))

        train = balancedData[:-200]
        test = balancedData[-200:]

        X = np.array([i[0] for i in train]).reshape(-1, self.proc_width,self.proc_height,1)
        Y = np.array([i[1] for i in train])

        test_X = np.array([i[0] for i in test]).reshape(-1, self.proc_width,self.proc_height,1)
        test_Y = np.array([i[1] for i in test])

        #if os.path.exists('{}/{}.tflearn'.format(model_dir,model_name)):
        #    model.load('{}/{}.tflearn'.format(model_dir,model_name))
        #elif os.path.exists('{}/checkpoint'.format(model_dir)):
        #    fName = open('{}/checkpoint'.format(model_dir),'r').readlines()[0].split(": ")[1][1:-2]
        #    model.load(fName)

        model.fit({'inputs':X},{'targets':Y}, n_epoch=n_epoch,
                  validation_set=({'inputs':test_X},{'targets':test_Y}),
                  snapshot_step=500, show_metric=True, run_id=run_id)
        model.save('{}/{}.tflearn'.format(model_dir,model_name))
