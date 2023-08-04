import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
import cv2
import datetime
import sys
import math

sys.path.append("/home/lemon/catkin_ws/src/aelos_smart_ros")
from leju import base_action  # NOQA:E402


class ImgConverter():
    def __init__(self):
        self.bridge = CvBridge()
        self.sub_chest = rospy.Subscriber(
            '/usb_cam_chest/image_raw', Image, self.cb_chest)
        self.sub_head = rospy.Subscriber(
            '/usb_cam_head/image_raw', Image, self.cb_head)
        self.img_chest = None
        self.img_head = None

    def cb_chest(self, msg):
        cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.img_chest = cv2_img

    def cb_head(self, msg):
        cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.img_head = cv2_img

    def chest_image(self):
        return self.img_chest

    def head_image(self):
        return self.img_head

################################## 动作执行####################################


def act(act_name):
    # time.sleep(1)
    base_action.action(act_name)


# 得到最大轮廓和对应的最大面积
def getAreaMaxContour(bin):  # 返回轮廓 和 轮廓面积
    """
    :param bin:二值图
    :return:
    """
    contours, _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_max_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(area_max_contour)
    if max_area < 25:  # 只有在面积大于25时，最大面积的轮廓才是有效的，以过滤干扰
        area_max_contour = None
        max_area = None
    return area_max_contour, max_area  # 返回最大的轮廓


def getlogtime():
    """
    获取当前时间(日、时)
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%d%H')
    return formatted_datetime


def getangle(point_l, point_r):
    """
    获得两个点连线的坡度角（opencv的坐标轴样式）
    :param point_l: 左侧点
    :param point_r: 右侧点
    :return: 坡度角（角度制）
    """
    angle = -math.atan((point_r[1]-point_l[1])/(point_r[0]-point_l[0]+1e-6))
    angle = angle*180/math.pi
    return angle

def getlen(points):
    """
    获得指定端点的长度
    :param points: 需要计算长度的两点
    """
    len = math.sqrt((points[0][1]-points[1][1])**2+(points[0][0]-points[1][0])**2)
    return len

def get_lowcon(contour):
    """
    获得轮廓重心y值
    :param contour: 需要进行查找的轮廓元组(tuple)
    """
    M = cv2.moments(contour)
    return M['m01']/(M['m00']+1e-6)
