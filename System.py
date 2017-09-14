import numpy as np
import cv2
import uinput
import time
import random
import keyboard
import MyUtils

class System:
    def __init__(self, screen_w, screen_h, screen_x, screen_y, proc_width, proc_height, mask_img, model):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.proc_width= proc_width
        self.proc_height= proc_height
        self.paused=False
        self.mask = cv2.imread(mask_img,cv2.IMREAD_GRAYSCALE)
        self.model = model
        if self.mask is None:
            print "Image mask not found"
            exit(1)
        elif self.mask.shape[1] != screen_w or self.mask.shape[0] != screen_h:
            print "Image mask has invalid size"
            exit(1)
        self.scc = MyUtils.ScreenCapture({'width': screen_w, 'top': screen_y, 'height': screen_h, 'left': screen_x})

        #cv2.namedWindow( "process", cv2.WINDOW_NORMAL )
        #cv2.resizeWindow('process', 600,600)
        #cv2.waitKey(1)

    def start(self):
        self.predictHistory = []
        self.paused=False

    def stop(self):
        pass

    def process(self):

        if keyboard.is_pressed('i'):
            time.sleep(0.4)
            if not keyboard.is_pressed('i'):
                self.paused = not self.paused
                if self.paused:
                    print ""
                    print "Paused controller"
                    keyboard.release('up')
                    keyboard.release('right')
                    keyboard.release('left')
                else:
                    print ""
                    print "Continue"

        if self.paused:
            time.sleep(0.01)
            return False

        img = self.scc.grab()
        #cv2.imshow("raw",  self.mask)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = cv2.Canny(img,100,200)
        img = cv2.bitwise_and(img,img,mask=self.mask)
        #cv2.imshow("process", img)

        img = cv2.resize(img,(self.proc_width,self.proc_height))
        y = self.model.predict([img.reshape(self.proc_width,self.proc_height,1)])[0]

        if len(self.predictHistory) > 3:
            self.predictHistory.pop(0)
        self.predictHistory.append(y)

        s = [0,0,0]
        for i in self.predictHistory:
            s += i
        s /= sum(s)

        names=["left","straight","right"]
        printWidth=80
        print "-"*(printWidth+40)
        print "Prediction: {}".format(y)
        for idx,i in enumerate(s):
            print ("{: <8}: {: <"+str(printWidth)+"} - {}%").format(names[idx],"|"*int(np.around(i*printWidth)),100*i )
        y = list(np.around(s))


        keyboard.release('up')
        keyboard.release('right')
        keyboard.release('left')

        if y == [1,0,0]:
            print "Action: left"
            keyboard.press('up')
            keyboard.press('left')
        elif y == [0,0,1]:
            print "Action: right"
            keyboard.press('up')
            keyboard.press('right')
        elif y == [0,1,0]:
            print "Action: straight"
            keyboard.press('up')
        else:
            print "Action: dont know ?!"

        print "-"*(printWidth+40)
        return True
