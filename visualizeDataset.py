import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_filename')
parser.add_argument('--isRawData',action='store_true')
args = parser.parse_args()
print args

trainData = list(np.load(args.data_filename))

cv2.namedWindow("trainImg", cv2.WINDOW_KEEPRATIO)

for d in trainData:
    img = d[0]
    keys = d[1]
    if args.isRawData:
        print "Keys: {}".format(keys)
    else:
        print "One hot: {}".format(keys)

    cv2.imshow("trainImg",img)
    cv2.resizeWindow('trainImg', 600,600)
    cv2.waitKey(1)
