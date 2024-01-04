import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
 
img_path = '' #填入图片所在文件夹的路径
img_Topath = '' #填入图片转换后的文件夹路径
img=cv2.imread("/home/yakki/Fast-SCNN-pytorch-master/2.png")
img = cv2.resize(img,(2048,1024))
cv2.imwrite("/home/yakki/Fast-SCNN-pytorch-master/3.png",img)


