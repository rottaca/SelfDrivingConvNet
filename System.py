import numpy as np
import cv2
import uinput
import time
import random
import keyboard

import MyUtils

class System:
    def __init__(self, screen_w, screen_h, screen_x, screen_y, mask_img):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.mask = cv2.imread(mask_img,cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            print "Image mask not found"
            exit(1)
        elif self.mask.shape[1] != screen_w or self.mask.shape[0] != screen_h:
            print "Image mask has invalid size"
            exit(1)
        self.scc = MyUtils.ScreenCapture({'width': screen_w, 'top': screen_y, 'height': screen_h, 'left': screen_x})


        cv2.namedWindow( "process", cv2.WINDOW_NORMAL )
        cv2.waitKey(1)

    def start(self):
        self.last_update = time.time()
        keyboard.press('up')

    def stop(self):
        keyboard.release('up')

    def process(self):
        img = self.scc.grab()
        #cv2.imshow("raw",  self.mask)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #img = cv2.Canny(img,100,200)
        img = cv2.bitwise_and(img,img,mask=self.mask)
        cv2.imshow("process", img)

        if time.time()-self.last_update > 2:
            self.last_update = time.time()
            x = random.randint(1,3)
            keyboard.press('up')

            if x == 1:
                keyboard.press('left')
                keyboard.release('right')
            elif x == 2:
                keyboard.press('right')
                keyboard.release('left')
            elif x == 3:
                keyboard.release('right')
                keyboard.release('left')
            else:
                exit(1)
