import rospy
import cv2
import utils
import os
import time
import threading
import itertools

rospy.init_node('picture')

ChestOrg_img = None  
HeadOrg_img = None  


#############更新图像#############
def updateImg():
    global ChestOrg_img, HeadOrg_img
    image_reader = utils.ImgConverter()
    while True:
        ChestOrg_img = image_reader.chest_image()
        HeadOrg_img = image_reader.head_image()
        time.sleep(0.05)

# 创建线程更新图像
th_capture = threading.Thread(target=updateImg)
th_capture.setDaemon(True)
th_capture.start()

if not os.path.exists('pics'):
    os.mkdir('pics')

while ChestOrg_img is None or HeadOrg_img is None:
    time.sleep(0.5)

counter = itertools.count(start=1,step=1)

for i in counter:
    cv2.imwrite('./pics/'+str(i)+'.jpg',ChestOrg_img)
    time.sleep(0.1)