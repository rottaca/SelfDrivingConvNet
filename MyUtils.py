import numpy as np
import uinput
import cv2
from PIL import Image
# Screen capturing
import mss

class ScreenCapture:
    def __init__(self, bbox=None):
        self.bbox = bbox
        self.sct = mss.mss()

    def grab(self):
        sct_img = self.sct.grab(self.bbox)
        img = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def get_nice_string(list_or_iterator):
    return ", ".join( str(x) for x in list_or_iterator)
