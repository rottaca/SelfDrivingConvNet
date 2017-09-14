import numpy as np
import cv2
import uinput
import time
import random
import keyboard
import os,sys

import tflearn

import MyUtils

class TrainingRecorder:

    def __init__(self, outputDir, keys, keys_to_onehot):
        self.outputDir = outputDir
        self.rawDataFileName = outputDir + "/rawData.npy"
        self.balancedDataFileName = outputDir + "/balancedData.npy"
        self.keys = keys
        self.keys_to_onehot = keys_to_onehot
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

    def setupDimensions(self,screen_w, screen_h, screen_x, screen_y, proc_width, proc_height):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.proc_width= proc_width
        self.proc_height= proc_height

    def start(self, interval):
        self.scc = MyUtils.ScreenCapture({'width': self.screen_w, 'top': self.screen_y, 'height': self.screen_h, 'left': self.screen_x})
        self.interval = interval
        self.last_time = time.time()
        self.trainData = []
        self.loadRawData()
        self.paused=False

    def save(self):
        print("Saving data of size: {}".format(len(self.trainData)))
        np.save(self.rawDataFileName,self.trainData)

    def process(self):

        if keyboard.is_pressed('i'):
            time.sleep(0.4)
            if not keyboard.is_pressed('i'):
                self.paused = not self.paused
                if self.paused:
                    print ""
                    print "Paused recording"
                else:
                    print ""
                    print "Continue"

        if self.paused:
            time.sleep(0.01)
            return False

        if time.time()-self.last_time > self.interval:
            self.last_time = time.time()
            img = self.scc.grab()
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(self.proc_width,self.proc_height))

            pressedKeys= []
            for k in self.keys:
                pressedKeys.append(int(keyboard.is_pressed(k)))

            if sum(pressedKeys) == 0:
                return True

            self.trainData.append([img,pressedKeys])
            return True

        return False

    def loadRawData(self):
        if os.path.exists(self.rawDataFileName):
            self.trainData = list(np.load(self.rawDataFileName))
            print "Loaded data with {} entries".format(len(self.trainData))
        else:
            print "No prerecorded data to load."

    def balanceData(self):
        random.shuffle(self.trainData)

        remapped_keys_and_imgs = [[] for i in range(len(self.keys_to_onehot))]
        skippedCount = 0
        print "Extracting data and remapping keys..."
        for idx, data in enumerate(self.trainData):
            img = data[0]
            keys = data[1]

            # Remap keys int
            one_hot = []
            typeIdx = 0
            for i, kList in enumerate(self.keys_to_onehot):
                # Pressed keys match any pattern ?
                if keys in kList:
                    one_hot.append(1)
                    typeIdx = i
                else:
                    one_hot.append(0)

            if sum(one_hot) != 1:
                print "One hot vector is empty! No keymapping found for keys: {}".format(keys)
                skippedCount+=1

            remapped_keys_and_imgs[typeIdx].append([img,one_hot])

        print "Done."
        print "Samples per type: " + str(map(len,remapped_keys_and_imgs))
        print "Skipped samples: " + str(skippedCount)
        minCnt = min(map(len,remapped_keys_and_imgs))
        print "Minimum samples per class: " + str(minCnt)
        print "Balancing..."
        
        # Merge data
        balancedData = [];
        for d in remapped_keys_and_imgs:
            balancedData.extend(d[:minCnt])

        print("Saving balanced of size: {}".format(len(balancedData)))
        random.shuffle(balancedData)
        np.save(self.balancedDataFileName,balancedData)
        print "Done."

    def trainModel(self, model, model_dir, model_name, run_id, n_epoch, mask_img):

        balancedData = list(np.load(self.balancedDataFileName))
        print "Loaded data with {} entries".format(len(balancedData))

        print "Masking data..."
        mask = cv2.imread(mask_img,cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print "Image mask not found"
            exit(1)
        elif mask.shape[1] != self.screen_w or mask.shape[0] != self.screen_h:
            print "Image mask has invalid size"
            exit(1)

        mask = cv2.resize(mask,(self.proc_width, self.proc_height))
        maskedData = []
        for img, key in balancedData:
            img = cv2.bitwise_and(img,img,mask=mask)
            maskedData.append([img,key])

        testSize = int(0.1*len(maskedData))
        train = maskedData[:-testSize]
        test = maskedData[-testSize:]

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
