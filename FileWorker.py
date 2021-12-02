import os
import cv2 as cv
import numpy as np

class FileWorker():
    def ReadAll(self,dir_path):
        paths=os.listdir(dir_path)
        return paths

    def ReadImg(self,path):
        extension=os.path.splitext( os.path.basename(path) )[1]
        if extension=='.jpg' or extension=='.png':
            img=cv.imread(path,0)
            return np.float32(img)

    def SaveImg(self,img,dir_path,name):
        cv.imwrite(os.path.join(dir_path,name+".png"),img)
    