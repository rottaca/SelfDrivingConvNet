import numpy as np
import cv2
import uinput
import time
import random
import keyboard
import os

import MyUtils

class TrainingRecorder:

    def __init__(self, screen_w, screen_h, screen_x, screen_y, interval, outputDir):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.scc = MyUtils.ScreenCapture({'width': screen_w, 'top': screen_y, 'height': screen_h, 'left': screen_x})
        self.interval = interval
        self.outputDir = outputDir
        self.rawDataFileName = outputDir + "/rawData.npy"
        self.balancedDataFileName = outputDir + "/balancedData.npy"
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
            img = cv2.resize(img,(80,60))

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
        self.trainData = list(np.load(self.rawDataFileName))
        print "Loaded data with {} entries".format(len(self.trainData))

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
            #cv2.resizeWindow('img', 600,600)

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

        print "fwd"
        time.sleep(3)
        for data in fwd:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(25)

        print "left"
        time.sleep(3)
        for data in left:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(25)

        print "right"
        time.sleep(3)
        for data in right:
            img = data[0]
            keys = data[1]
            cv2.imshow("img", img)
            cv2.resizeWindow('img', 600,600)
            cv2.waitKey(25)


        balancedData = fwd + left + right
        print("Saving balanced of size: {}".format(len(balancedData)))
        random.shuffle(balancedData)
        np.save(self.balancedDataFileName,balancedData)
        print "Done."
