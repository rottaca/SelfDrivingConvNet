import numpy as np
import cv2
import uinput
import time
import random
import keyboard
import MyUtils

class System:
    def __init__(self, screen_w, screen_h, screen_x, screen_y, proc_width, proc_height, mask_img, model, keys, keys_to_onehot, onehot_names):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.proc_width= proc_width
        self.proc_height= proc_height
        self.keys = keys
        self.keys_to_onehot = keys_to_onehot
        self.onehot_names = onehot_names
        self.paused=False
        self.mask = cv2.imread(mask_img,cv2.IMREAD_GRAYSCALE)
        self.model = model
        if self.mask is None:
            print "Image mask not found"
            exit(1)
        elif self.mask.shape[1] != proc_width or self.mask.shape[0] != proc_height:
            print "Image mask has invalid size"
            exit(1)
        self.scc = MyUtils.ScreenCapture({'width': screen_w, 'top': screen_y, 'height': screen_h, 'left': screen_x})

        cv2.namedWindow("input", cv2.WINDOW_NORMAL )
        #cv2.resizeWindow('input', 600,600)

    def start(self):
        self.y = []
        self.paused=False

    def stop(self):
        pass

    def process(self):

        if keyboard.is_pressed('i'):
            time.sleep(0.2)
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
            
        t = time.time()
        img = self.scc.grab()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(self.proc_width,self.proc_height))
        img = cv2.bitwise_and(img,img,mask=self.mask)
        #print "dt capture: {}".format(time.time()-t)
        cv2.imshow("input",img)
        cv2.waitKey(1)
        #t = time.time()
        # Predict model output for a single sample
        y = self.model.predict([img.reshape(self.proc_width,self.proc_height,1)])[0]
        #print "dt predict: {}".format(time.time()-t)
        
       # if len(self.y) > 0:
       #     y = self.y*0.1+0.9*y
       # self.y = y
        
        # Print prediction as number and as pretty bars
        printWidth=40
        print "-"*(printWidth+30)
        print "Prediction: {}".format(y)
        for idx,i in enumerate(y):
            print ("{: <8}: {: <"+str(printWidth)+"} - {}%").format(
                   self.onehot_names[idx],"|"*int(np.around(i*printWidth)),100*i )

        # If confidence is below 50 %, y is zero everywere
        # Don't do anything
        if max(y) < 0.5:
            keyOutput = self.keys_to_onehot[0][0]
            print "Action: dont know ?!"
        else:
            # Map one hot vector back to key combination
            # Use first key combination for this type of steering as output
            keyOutput = self.keys_to_onehot[np.argmax(y)][0]
            # Print name of action
            print "Action: {}".format(self.onehot_names[np.argmax(y)])
        
        # Release all keys
        #for k in self.keys:

        # Press keys and print key names
        print "Pressed keys:"
        for idx, k in enumerate(keyOutput):
            # Press key at position idx ?
            if k == 1:
                keyboard.press(self.keys[idx])
                print self.keys[idx]
            else:
                keyboard.release(self.keys[idx])

        print "-"*(printWidth+30)

        return True
