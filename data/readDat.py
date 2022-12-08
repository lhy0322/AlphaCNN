import random
import cv2
import os

imgPath = 'tiffphoto/'
imgList = os.listdir(imgPath)

for img in imgList:

    orimgPath = imgPath + img
    orimg = cv2.imread(orimgPath, -1)
    x = random.randint(500, 700)
    y = random.randint(1400, 1600)
    split_1 = orimg[y-300:y+300+1, x-300:x+300+1]
    split_2 = orimg[500-300:500+300+1, 600-300:600+300+1]
    cv2.imwrite("DATA/" + img[0:-5] + "_1.tiff", split_1)
    cv2.imwrite("DATA/" + img[0:-5] + "_2.tiff", split_2)

