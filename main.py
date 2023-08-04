#!/usr/bin/env python3
# coding:utf-8

from re import I
import numpy as np
import cv2
import math
import threading
import time
import rospy
import sys
import utils
import statistics
from enum import Enum

chest_width = 480
chest_height = 640
head_width = 640
head_height = 480
ChestOrg_img = None  # 原始图像更新
HeadOrg_img = None  # 原始图像  更新
t = 0   # 用时

Debug = False

############ 颜色阈值 #############
start_door_color_range = {
    'yellow_door': [(22, 124, 77), (52, 255, 255)],
    'black_door': [(0, 0, 0), (179, 67, 90)],
}

end_door_color_range = {
    'yellow_door': [(28, 90, 90), (34, 255, 255)],
    'black_door': [(0, 0, 0),(180, 80, 80)],
}

hole_color_range = {
    'green_hole_chest': [(57, 94, 0), (89, 255, 230)],
    'blue_hole_chest': [(102, 123, 132), (110, 213, 235)],
}

landmine_color_range = {
    # 'blue_baf': [(106, 70, 42), (255, 255, 200)],
    # 'blue_baf_head': [(85, 51, 32), (126, 255, 211)],
    # 'blue_baf_chest': [(89, 80, 65), (129, 255, 255)],
    # 'black_dir': [(0, 0, 0), (180, 60, 60)],

    'blue_baf_head': [(104, 70, 43), (118, 195, 229)],     #night
    'blue_baf_chest': [(105, 78, 54), (135, 203, 173)],
    'black_dir': [(0, 0, 0), (180, 60, 60)],

    # 'blue_baf_head': [(107, 51, 32), (126, 255, 211)],     #afternoon
    # 'blue_baf_chest': [(89, 80, 65), (129, 255, 255)],
    # 'black_dir': [(0, 0, 0), (180, 60, 60)],
}


bluedoor_color_range = {
    'blue_door_chest':[(87,111,0),(122,255,255)],
    'green_bridge':[(40,65,0),(84,255,255)]
}


bridge_color_range = [(57, 94, 0), (89, 255, 230)]



stair_color_range = {
    'blue_floor': [(100, 100, 120), (115, 255, 255)],
    'green_floor': [(57, 107, 15), (97, 255, 213)],
    'red_floor': [(0, 133, 140), (180, 255, 255)],
}


############# 更新图像#############
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

###########################################################################
##########                      起点门                            ##########
###########################################################################

def start_door():
    crossbardownalready = False
    PERCENT_THRESH = 5
    intercept = 500
    global HeadOrg_img, t
    t = cv2.getTickCount()
    goflag = 0
    while True:
        if goflag:
            utils.act("fastForward05")
            utils.act("Stand")
            print("开启下一关")

            # utils.act("forwardSlow0403")

            # utils.act("fast_forward_step")
            # cv2.destroyAllWindows()
            break
        else:  # 判断门是否抬起
            handling = HeadOrg_img.copy()

            border = cv2.copyMakeBorder(handling, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))  # 扩展白边，防止边界无法识别
            handling = cv2.resize(border, (640, 480), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
            frame_gauss = cv2.GaussianBlur(handling, (21, 21), 0)  # 高斯模糊
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间

            frame_hsv = frame_hsv[0:480, 0:intercept]  # 裁剪掉图像右边缘部分

            frame_door_yellow = cv2.inRange(frame_hsv, start_door_color_range['yellow_door'][0],
                                            start_door_color_range['yellow_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            frame_door_black = cv2.inRange(frame_hsv, start_door_color_range['black_door'][0],
                                           start_door_color_range['black_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            frame_door = cv2.add(frame_door_yellow, frame_door_black)

            open_pic = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 开运算 去噪点
            closed_pic = cv2.morphologyEx(open_pic, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))  # 闭运算 封闭连接
            (contours, hierarchy) = cv2.findContours(closed_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            areaMaxContour, area_max = utils.getAreaMaxContour1(contours)  # 找出最大轮廓
            percent = round(100 * area_max / (640 * 480), 2)  # 最大轮廓的百分比

            if Debug:
                cv2.line(closed_pic, [0, intercept], [640, intercept], (100, 255, 100), 1)
                if percent > PERCENT_THRESH:
                    cv2.putText(closed_pic, percent, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(closed_pic, percent, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imwrite('./closed_pic.jpg', closed_pic)  # 查看识别情况

            # 根据比例得到是否前进的信息
            if percent > PERCENT_THRESH:
                crossbardown = True
            else:
                crossbardown = False


            if not crossbardownalready:
                if crossbardown:
                    crossbardownalready = True
                    print("横杆已落下，等待横杆开启")
                    print('percent = ', percent)
                else:
                    print("横杆未落下，先等待横杆落下")
                    print('percent = ', percent)
            else:
                if not crossbardown:
                    goflag = True
                    print("机器人启动")
                else:
                    print("横杆已落下，等待横杆开启")
            time.sleep(0.1)

    return goflag


###########################################################################
##########                     过坑部分                           ##########
###########################################################################

angle_bias = 0


def get_robust_angle_hole(app_e, threshold):
    """
    获取远处底线角度
    :param app_e: 拟合多边形程度
    :param threshold 颜色阈值
    :return: 角度值，正值应左转，负值应右转
    """
    angles = []
    # 获取多张照片
    for _ in range(10):
        if ChestOrg_img is not None:
            img = ChestOrg_img.copy()
            img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, threshold[0], threshold[1])

            mask[:, 500:] = 0  # 处理相机畸变带来的误判

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 5000:
                polyappro = cv2.approxPolyDP(
                    max_cnt, epsilon=app_e*cv2.arcLength(max_cnt, closed=True), closed=True)
                sorted_poly = sorted(np.squeeze(polyappro),
                                     key=lambda x: -x[1], reverse=True)
                if len(sorted_poly) >= 2:
                    topleft = min(sorted_poly, key=lambda x: 0.3*x[0]+0.7*x[1])
                    topright = None
                    for point in sorted_poly:
                        if point[0] != topleft[0] or point[1] != topleft[1]:
                            topright = point
                            break
                    angle = utils.getangle(topleft, topright)
                    angles.append(angle)
                    if Debug:
                        debug = cv2.line(img.copy(), tuple(topleft), tuple(
                            topright), (0, 0, 255), thickness=3)
                        debug = cv2.circle(debug, tuple(
                            topleft), 3, (255, 0, 0), -1)
                        green = cv2.bitwise_and(
                            img.copy(), img.copy(), mask=mask)
                        cv2.imwrite('./log/hole/'+utils.getlogtime() +
                                    'angle_correction_line.jpg', debug)
                        cv2.imwrite('./log/hole/'+utils.getlogtime() +
                                    'angle_correction_green.jpg', green)
                else:
                    print('拟合多边形边数小于2')

            time.sleep(0.05)  # 等待获取下一张图片
    # 取中位数，确保鲁棒性
    if len(angles):
        angle = statistics.median(angles)
        return angle


def get_horizonal_position_hole(app_e, frame_middle_x, threshold):
    if ChestOrg_img is not None:
        img = ChestOrg_img.copy()
        img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, threshold[0], threshold[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            contours = filter(lambda x: cv2.contourArea(x) > 1000, contours)
            max_cnt = max(contours, key=cv2.contourArea)
            polyappro = cv2.approxPolyDP(
                max_cnt, epsilon=app_e*cv2.arcLength(max_cnt, closed=True), closed=True)
            leftbottom = max(polyappro, key=lambda x: -x[0][0]+x[0][1])
            dist = frame_middle_x - leftbottom[0][0]
            if Debug:
                debug = cv2.circle(img, tuple(
                    leftbottom[0]), 3, (0, 0, 255), -1)
                cv2.imwrite('./log/hole/'+utils.getlogtime()+'pos.jpg', debug)
            return dist


def find_max_area(threshold):
    """
    根据阈值计算最大面积
    :param threshold:
    :return: 面积
    """
    if ChestOrg_img is not None:
        img = ChestOrg_img.copy()
        img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, threshold[0], threshold[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            max_cnt = max(contours, key=cv2.contourArea)
            max_area = cv2.contourArea(max_cnt)
            return max_area


def pass_hole(threshold):
    """
    过坑
    :param threshold:地面颜色阈值
    """
    print('进入过坑关')
    orintation_right = False
    horizonal_right = False
    while True:
        print('#######################################################')
        if ChestOrg_img is None:
            print('相机未准备好')
            time.sleep(1)
            continue
        area = find_max_area(threshold)
        angle = get_robust_angle_hole(0.005, threshold)
        print('当前绿色面积：', area)
        if area is None or area < 20000:
            print('往前走通过本关')
            utils.act('panR1')
            utils.act('panR1')
            utils.act('panR1')
            utils.act('panR1')
            break

        if angle is None:
            print('没找到角度，后退一点')
            utils.act('Backward0')
            continue
        print('当前朝向：', angle+angle_bias)
        if -3 < angle+angle_bias < 3:
            print('朝向正确')
            orintation_right = True
        elif angle+angle_bias <= -3:
            orintation_right = False
            if angle+angle_bias < -5 and area > 70000:
                print('大右转')
                utils.act('turnR1')
            else:
                print('小右转')
                utils.act('turnR0')
        elif angle+angle_bias >= 3:
            orintation_right = False
            if angle+angle_bias > 5 and area > 70000:
                print('大左转')
                utils.act('turnL1_1')
            else:
                print('小左转', angle)
                utils.act('turnL0')

        if orintation_right:  # 朝向正确，检查左右偏移
            pos = get_horizonal_position_hole(0.005, 320, threshold)
            print('左边边界位置:', pos)
            if 110 < pos < 190:
                horizonal_right = True
            if pos <= 110:
                horizonal_right = False
                print('右移')
                utils.act('panR1')
            if pos >= 190:
                horizonal_right = False
                if pos < 230:
                    print('小左移')
                    utils.act('panL0')
                else:
                    print('大左移')
                    utils.act('panL1_1')

        if orintation_right and horizonal_right:
            print('向前走')
            utils.act('Forward1')


# 识别过坑关卡
# 过坑识别
def hole_recognize(color):
    src = ChestOrg_img.copy()
    Area = 0
    src = src[int(100):int(400), int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_img, hole_color_range[color][0], hole_color_range[color][1])
    closed = cv2.dilate(mask, None, iterations=5)
    closed = cv2.erode(closed, None, iterations=8)

    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area = max(contours, key=cv2.contourArea)
        Area = cv2.contourArea(max_area)
        rect = cv2.minAreaRect(max_area)
        # print(rect[0])
        # # print(Area)
    contours2, hierarchy2 = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if Area > 20000 and len(contours2) >= 2:
        return True
    else:
        return False

#######################################################################
####################            地雷               ####################
#######################################################################

def bottom_polydp_and_points(frame,color):

    def centre(contour):
        M = cv2.moments(contour)
        return M['m01'] / (M['m00'] + 1e-6)

    Imask = cv2.inRange(frame, landmine_color_range[color][0], landmine_color_range[color][1])

    mask = Imask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)

    cv2.imwrite('./log/landmine/'+utils.getlogtime()+'bluepart.jpg', cv2.bitwise_and(frame, frame, mask=mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出所有轮廓
    
    adapting_threshold = 500  # 自适应阈值
    while True:
        # 筛选轮廓
        contours_filtered = list(filter(lambda x: cv2.contourArea(x) > adapting_threshold, contours))
        if len(contours_filtered) >= 2:
            break
        adapting_threshold -= 50
        if adapting_threshold < 200:
            print('没有合适的蓝色轮廓')
            return None, None, None, None


    cnt = max(contours_filtered, key=centre)  # 最靠下的轮廓
    cnt = np.squeeze(cnt)

    polydp = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

    bottom_left = max(cnt, key=lambda x: -x[0] + 3 * x[1])
    bottom_right = max(cnt, key=lambda x: x[0] + x[1])

    return polydp, bottom_right, bottom_left, mask


def obstacle():
    global HeadOrg_img, step, ChestOrg_img, state_sel
    state_sel = 'obstacle'
    print("/-/-/-/-/-/-/-/-/-进入obstacle")
    step = 0  # 用数字表示在这一关中执行任务的第几步

    print("进入地雷阵")

    DIS_SWITCH_CAM = 300
    DIS_PREPARE_FOR_ROLL = 370
    recog = True
    error = 0
    cam_in_use = 'chest'

    begin_adjust = True
    angle_thresh = 4

    step_lei = 0
    cnt_lei = 0

    lei_para = {
        'dis': [325, 358],  # 开始缓慢靠近，不能（不用）再靠近
        'lr': [152, 210, 320, 430, 488],
        'exclude': [250, 460, 120, 520],  # 前后左右
    }

    while (1):
        print('####################################################')
        Chest_img = ChestOrg_img.copy()
        Head_img = HeadOrg_img.copy()

        Chest_hsv = cv2.cvtColor(Chest_img, cv2.COLOR_BGR2HSV)
        Chest_hsv = cv2.GaussianBlur(Chest_hsv, (3, 3), 0)
        Head_hsv = cv2.cvtColor(Head_img, cv2.COLOR_BGR2HSV)
        Head_hsv = cv2.GaussianBlur(Head_hsv, (3, 3), 0)

        c_bottom_poly, c_bottom_right, c_bottom_left, mask_chest = bottom_polydp_and_points(Chest_hsv, 'blue_baf_chest')
        h_bottom_poly, h_bottom_right, h_bottom_left, mask_head = bottom_polydp_and_points(Head_hsv, 'blue_baf_head')

        if c_bottom_poly is not None:
            bottom_dis = (c_bottom_right[1] + c_bottom_left[1]) / 2  # 用胸部摄像头得到的bottom_dis判断挡板距离
            print("bottom_dis=", bottom_dis)

            # bottom_dis大时用胸部摄像头，小时用头部摄像头
            if bottom_dis > DIS_SWITCH_CAM:
                bottom_angle = -math.atan(
                    (c_bottom_right[1] - c_bottom_left[1]) /
                    (c_bottom_right[0] - c_bottom_left[
                        0])) * 180.0 / math.pi  # negative signal comes from the direction of y-axis
                bottom_center = (c_bottom_right + c_bottom_left) / 2
                print("使用胸部摄像头校正，bottom_angle = ", bottom_angle)
                cam_in_use = 'chest'
                angle_thresh = 5
            elif h_bottom_poly is not None:
                bottom_angle = -math.atan(
                    (h_bottom_right[1] - h_bottom_left[1]) /
                    (h_bottom_right[0] - h_bottom_left[0])) * 180.0 / math.pi
                bottom_center = (h_bottom_right + h_bottom_left) / 2
                print("使用头部摄像头校正，bottom_angle = ", bottom_angle)
                cam_in_use = 'head'
                angle_thresh = 4

        if c_bottom_poly is not None or h_bottom_poly is not None:

            # 防止摄像头一直识别不到挡板机器人不动，一般不会发生
            if bottom_angle == 91 and recog:
                print("摄像头识别错误，前进一小步")
                utils.act('Forward0')
                error += 1
                if error > 2:
                    print("一直识别不到挡板，先进入过雷阶段再说")
                    recog = False
                    begin_adjust = False
                continue

            # 入地雷阵前调整，只调角度，不调位置
            if begin_adjust:
                print('即将进入地雷关，先矫正角度，使用头部摄像头')
                if bottom_angle > angle_thresh:
                    print("bottom_angle角度=", bottom_angle)
                    print("往右偏了，需要左转矫正")
                    utils.act('turnL0')
                    time.sleep(0.2)
                elif bottom_angle < -angle_thresh:
                    print("bottom_angle角度=", bottom_angle)
                    print("往左偏了，需要右转矫正")
                    utils.act('turnR0')
                    time.sleep(0.2)
                    continue
                else:
                    print("OK了，bottom_angle角度=", bottom_angle)
                    begin_adjust = False

            # 挡板调整
            if bottom_dis > DIS_PREPARE_FOR_ROLL:  # 距离挡板很近了，开始挡板调整
                print("bottom_dis>%.2f, bottom_dis=%.2f" % (DIS_PREPARE_FOR_ROLL, bottom_dis), "雷阵结束，开始挡板调整")
                return True
            else:
                print("bottom_dis不足继续地雷识别")
                pass

            # 太歪的时候要调整
            if cam_in_use == 'head':
                if bottom_angle < -7 and bottom_center[0] > 460:
                    print("往左偏，危险！修正后避雷不能左移了")
                    cnt_lei += 15
                    utils.act('turnR1')
                    time.sleep(0.2)
                elif bottom_angle > 7 and bottom_center[0] < 140 and bottom_angle < 90:  # bottom_angle = 91 是没识别到挡板
                    print("往右偏，危险！修正后避雷不能右移了")
                    cnt_lei -= 15
                    utils.act('turnL1')
                    time.sleep(0.2)
            else:
                if bottom_angle < -7 and bottom_center[0] > 430:  ###### 机器人来了之后记得拍照片修改数值
                    print("往左偏，危险！修正后避雷不能左移了")
                    cnt_lei = 15
                    utils.act('turnR1')
                    time.sleep(0.2)
                elif bottom_angle > 7 and bottom_center[0] < 180 and bottom_angle < 90:
                    print("往右偏，危险！修正后避雷不能右移了")
                    cnt_lei = -15
                    utils.act('turnL1')
                    time.sleep(0.2)
        else:
            print("头部与胸部摄像头都识别不到轮廓，需要调整阈值！")

        # 以下地雷检测
        hsv = cv2.cvtColor(Chest_img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask_lei = cv2.inRange(hsv, landmine_color_range['black_dir'][0], landmine_color_range['black_dir'][1])
        Imask_lei = cv2.erode(Imask_lei, None, iterations=3)
        Imask_lei = cv2.dilate(Imask_lei, np.ones((3, 3), np.uint8), iterations=2)
        cv2.imwrite('./Imask_lei.jpg', Imask_lei)  # 二值化后图片显示
        contours, hierarchy = cv2.findContours(Imask_lei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
        print("contours lens:", len(contours))
        cv2.drawContours(Chest_img, contours, -1, (255, 0, 255), 2)
        cv2.imwrite('./Corg_img_Imask_lei.jpg', Chest_img)

        left_point = [640, 0]
        right_point = [0, 0]

        if len(contours) != 0:

            Big_battle = [0, 0]

            for c in contours:
                rect = cv2.minAreaRect(c)  # 最小外接矩形
                box = cv2.boxPoints(rect)  # 我们需要矩形的4个顶点坐标box, 通过函数 cv2.cv.BoxPoints() 获得
                box = np.intp(box)  # 最小外接矩形的四个顶点
                box_Ax, box_Ay = box[0, 0], box[0, 1]
                box_Bx, box_By = box[1, 0], box[1, 1]
                box_Cx, box_Cy = box[2, 0], box[2, 1]
                box_Dx, box_Dy = box[3, 0], box[3, 1]
                box_centerX = int((box_Ax + box_Bx + box_Cx + box_Dx) / 4)
                box_centerY = int((box_Ay + box_By + box_Cy + box_Dy) / 4)
                box_center = [box_centerX, box_centerY]
                # cv2.circle(Chest_img, (box_centerX,box_centerY), 7, (0, 255, 0), -1) #距离比较点 绿圆点标记
                # cv2.drawContours(Chest_img, [box], -1, (255,0,0), 3)

                # 剔除图像上部分点 和底部点
                if box_centerY < lei_para['exclude'][0] or box_centerY > lei_para['exclude'][1]:
                    continue

                # 遍历点 画圈
                if Debug:
                    cv2.circle(Chest_img, (box_centerX, box_centerY), 8, (0, 0, 255), 2)  # 圆点标记识别黑点
                    cv2.imwrite('./Chest_img.jpg', Chest_img)

                # 找出最左点与最右点
                if box_centerX < left_point[0]:
                    left_point = box_center
                if box_centerX > right_point[0]:
                    right_point = box_center

                if box_centerX <= lei_para['exclude'][2] or box_centerX >= lei_para['exclude'][
                    3]:  # 排除左右边沿点 box_centerXbox_centerX 240
                    continue
                if math.pow(box_centerX - 300, 2) + math.pow(box_centerY - 480, 2) < math.pow(Big_battle[0] - 300,
                                                                                              2) + math.pow(
                    Big_battle[1] - 480, 2):
                    Big_battle = box_center  # 这个是要规避的黑点

            # 显示图
            if Debug:
                cv2.circle(Chest_img, (left_point[0], left_point[1]), 7, (0, 255, 0), -1)  # 圆点标记
                cv2.circle(Chest_img, (right_point[0], right_point[1]), 7, (0, 255, 255), -1)  # 圆点标记
                cv2.circle(Chest_img, (Big_battle[0], Big_battle[1]), 7, (255, 255, 0), -1)  # 圆点标记
                cv2.putText(Chest_img, "botton_angle: " + str(int(bottom_angle)), (230, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(Chest_img, "bottom_center_x:" + str(int(bottom_center[0])), (230, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(Chest_img, "bottom_center_y:" + str(int(bottom_center[1])), (230, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(Chest_img, "Big_battle x,y:" + str(int(Big_battle[0])) + ', ' + str(int(Big_battle[1])),
                            (230, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.line(Chest_img, (Big_battle[0], Big_battle[1]), (240, 640), (0, 255, 255), thickness=2)

                cv2.line(Chest_img, (0, 500), (480, 500), (255, 255, 255), thickness=2)
                cv2.imwrite('./Chest_img.jpg', Chest_img)

            if step_lei == 0:
                if Big_battle[1] < lei_para['dis'][0]:
                    print("前进靠近 Forward1 ", Big_battle[1])
                    utils.act("Forward1")
                    if bottom_dis > DIS_PREPARE_FOR_ROLL - 50:
                        print("很靠近挡板了，如果有雷就先避一下")
                        if cnt_lei > 15:  # 净左移超过5步
                            step_lei = 1
                        elif cnt_lei < -15:
                            step_lei = 2  # 净右移超过5步
                        else:
                            step_lei = 3
                        continue
                elif Big_battle[1] < lei_para['dis'][1]:
                    print("慢慢前进靠近 Forward0", Big_battle[1])
                    utils.act("Forward0")
                    if bottom_dis > DIS_PREPARE_FOR_ROLL - 50:
                        print("很靠近挡板了，如果有雷就先避一下")
                        if cnt_lei > 15:  # 净左移超过5步
                            step_lei = 1
                        elif cnt_lei < -15:
                            step_lei = 2  # 净右移超过5步
                        else:
                            step_lei = 3
                        continue
                else:
                    if cnt_lei > 15:  # 净左移超过5步
                        step_lei = 1
                    elif cnt_lei < -15:
                        step_lei = 2  # 净右移超过5步
                    else:
                        step_lei = 3

            elif step_lei == 1:  # 只能右移
                print("step_lei=1, 只能右移")
                if lei_para['lr'][0] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][1]:
                    print("右移一点避雷 panR0", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panR0")
                    cnt_lei -= 1
                elif lei_para['lr'][1] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][2]:
                    print("右移一步避雷 panR1", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panR1")
                    cnt_lei -= 4
                elif lei_para['lr'][2] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][4]:
                    print("右移两步避雷 panR1*2", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panR1")
                    utils.act("panR1")
                    cnt_lei -= 8
                else:
                    if bottom_dis >= DIS_PREPARE_FOR_ROLL - 50:
                        print("很靠近挡板了，只能前进一小步，然后转入挡板关")
                        utils.act('Forward0')
                        time.sleep(0.2)
                        return True
                    print("不在调整范围，前进")
                    utils.act("Forward1")
                    time.sleep(0.05)
                    step_lei = 0

            elif step_lei == 2:  # 只能左移
                print("step_lei=2, 只能左移")
                if lei_para['lr'][3] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][4]:
                    print("左移一点避雷 panL0", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panL0")
                    cnt_lei += 1
                elif lei_para['lr'][2] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][3]:
                    print("左移一步避雷 panL1", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panL1")
                    cnt_lei += 4
                elif lei_para['lr'][0] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][2]:
                    print("左移两步避雷 panL1*2", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panL1")
                    utils.act("panL1")
                    cnt_lei += 8
                else:
                    if bottom_dis >= DIS_PREPARE_FOR_ROLL - 50:
                        print("很靠近挡板了，只能前进一小步，然后转入挡板关")
                        utils.act('Forward0')
                        time.sleep(0.2)
                        return True
                    print("不在调整范围，前进")
                    utils.act("Forward1")
                    time.sleep(0.05)
                    step_lei = 0

            elif step_lei == 3:
                print("step_lei=3")
                if (lei_para['lr'][0] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][1]):
                    print("右移一点避雷 panR0", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panR0")
                    cnt_lei -= 1
                elif (lei_para['lr'][1] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][2]):
                    print("右移一步避雷 panR1", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panR1")
                    cnt_lei -= 4
                elif (lei_para['lr'][2] <= Big_battle[0] and Big_battle[0] < lei_para['lr'][3]):
                    print("向左移一步避雷 panL0", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panL0")
                    cnt_lei += 1
                elif (lei_para['lr'][3] <= Big_battle[0] < lei_para['lr'][4]):
                    print("向左移一点避雷 panL1", Big_battle[0])
                    utils.act("Stand")
                    utils.act("panL1")
                    cnt_lei += 4
                else:
                    if bottom_dis >= DIS_PREPARE_FOR_ROLL - 50:
                        print("很靠近挡板了，只能前进一小步，然后转入挡板关")
                        utils.act('Forward0')
                        time.sleep(0.2)
                        return True
                    print("不在调整范围，前进")
                    utils.act("Forward1")
                    time.sleep(0.05)
                    step_lei = 0
        else:
            print("未识别到雷，继续向前")
            utils.act("Forward1")
            time.sleep(0.05)

    return True


########################################################################
##################               翻挡板              ####################
########################################################################

range_pos_dangban = [110,530,385,300,340]# 01左右端点合适值 2中点开始翻临界值 34左右移动中心点边界
dangban_color = [(62,51,0),(132,250,255)]

def findlow_dangban(contours,key=cv2.contourArea,rt_cnt=False):
    """
    胸部摄像头，找指定key下最大轮廓的最低边线
    :param contours: 轮廓
    :param key: 比较函数，用于筛选符合条件的轮廓
    :param rt_cnt: 是否返回所在轮廓标志位
    :return: 底线端点(及其所在轮廓)
    """
    def compare(points):
        mediumy = (points[0][1]+points[1][1])/2
        len = math.sqrt((points[0][1]-points[1][1])**2+(points[0][0]-points[1][0])**2)
        angle = abs(utils.getangle(points[0],points[1]))
        comp = 0.6*mediumy+0.2*len+0.2*(-angle)
        return comp
    max_contour = max(contours, key=key)
    poly = cv2.approxPolyDP(max_contour, 0.005 * cv2.arcLength(max_contour, True), True)
    line = []
    for i in range(len(poly)):
        line.append((np.squeeze(poly[i - 1]), np.squeeze(poly[i])))
    line = list(filter(lambda x:abs(utils.getangle(x[0],x[1]))<40,line))
    line = sorted(line,key=compare,reverse=True)
    loi = line[0]
    if rt_cnt is False:
        return loi
    else: return loi,max_contour


def dangban():
    while True:
        if ChestOrg_img is not None:
            img = ChestOrg_img.copy()
            img = cv2.resize(img, (640, 480))
            img[:, 0:80] = 0
            img[:, 590:640] = 0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            kernal = 5
            iteration = 1
            mask = cv2.inRange(img_hsv, dangban_color[0], dangban_color[1])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernal, kernal)), iterations=iteration)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernal, kernal)), iterations=iteration)

            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            while True:
                loi, cnt = findlow_dangban(contours, key=utils.get_lowcon, rt_cnt=True)
                len_loi = utils.getlen(loi)

                if Debug:
                    img_line = cv2.line(img.copy(), tuple(loi[0]), tuple(loi[1]), (0, 255, 0), 3)
                    cv2.imwrite('img_line.jpg', img_line)
                    print('太短了', len_loi)

                if len_loi < 150:
                    cnt_temp = tuple(e for e in contours if not np.any(e == cnt))
                    contours = cnt_temp
                    continue
                else:
                    break

            angle = utils.getangle(loi[0], loi[1]) - 3
            medium_pos = (int((loi[0][0] + loi[1][0]) / 2), int((loi[0][1] + loi[1][1]) / 2))
            pos_flag = False
            ########################
            if Debug:
                debug = ChestOrg_img.copy()
                cv2.line(debug, tuple(loi[0]), tuple(loi[1]), (0, 0, 255), 3)
                print(loi)
                cv2.imwrite('dangban.jpg', debug)
                cv2.imwrite('mask.jpg', mask)
            ####################
            if abs(loi[0][1] - range_pos_dangban[0]) <= 30 and abs(loi[1][1] - range_pos_dangban[1]) <= 30 and abs(
                    angle) <= 5:
                print('左右位置合适,向前怼两步,开始翻墙')
                utils.act('Forward0')
                utils.act('Forward0')
                utils.act('RollRail')
                pos_flag = True

            if medium_pos[1] > range_pos_dangban[2]:
                print('向前怼一步，开始翻墙')
                utils.act('Forward0')
                utils.act('RollRoil')
                break

            if angle > 4:
                print('向左转：', angle)
                utils.act('turnL0')
            elif angle < -4:
                print('向右转：', angle)
                utils.act('turnR0')
            elif pos_flag == False:
                print('对正了')
                if medium_pos[0] < range_pos_dangban[3]:
                    if medium_pos[0] < range_pos_dangban[3] - 10:
                        utils.act('panL1')
                        continue
                    print('向左移')
                    utils.act('panL0')
                elif medium_pos[0] > range_pos_dangban[4]:
                    if medium_pos[0] > range_pos_dangban[4] + 10:
                        utils.act('panR1')
                    print('向右移')
                    utils.act('panR1')
                else:
                    print('左右位置正确')
                    if medium_pos[1] < range_pos_dangban[2]:
                        print('向前走两步')
                        utils.act('Forward0')
                        utils.act('Forward0')
                    else:
                        print('向前怼一步，开始翻墙')
                        utils.act('Forward0')
                        utils.act('RollRail')
                        break

########################################################################
##################                过门               ####################
########################################################################

"""
pos_set
angle_set
需要调

现在假定下一关是独木桥
    如果下一关是楼梯
        1.仍然使用绿色掩膜，中点边界阈值统一上调15
        2.进入下一关中点边界上调15
    如果下一关是回型桥
        把上一关的阈值设的苛刻一点，边界缩水14
        角度阈值不变
"""
angle_change = 1
pos_change_max = 20
pos_change_min = 15
pos_set = {
    'step1':[185,220,170,235,220],
            # 01正常区间边界 23急需前后调整边界 4急需右移边界
    'step2':[392,408,380,420,400,250,379],
            # 01头部远端桥中点正常边界
            # 23头部头部远端桥中点急需前后调整边界
            # 4胸部摄像头右侧点继续右移边界
            # 5切换胸部摄像头差值条件
            # 6进入下一关中点边界
    'step3':[323,403,313,413,305],
            # 01正常区间边界 23急需前后调整边界 4通关右下角点两次均值边界条件
}
angle_set = {
    # 每一步角度阈值和角度偏置
    'step1':[4,0.8],
    'step2':[2.5,10,13.3],
    'step3':[4,0.8]
}

def get_robust_angle(app_e, threshold):
    """
    头部摄像头，获得指定hsv下底边线
    :param app_e: 多边形近似程度
    :param threshold: 掩膜范围
    :return: 角度，两个端点，右侧连线与底边框交点
    """
    angles = []
    botposes = []
    botlefts = [[], []]
    botrights = [[], []]
    # 获取多张照片
    for _ in range(5):
        if HeadOrg_img is not None:
            img = HeadOrg_img.copy()
            img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, threshold[0], threshold[1])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 1000:
                polyappro = cv2.approxPolyDP(max_cnt, epsilon=app_e * cv2.arcLength(max_cnt, closed=True), closed=True)
                sorted_poly = sorted(np.squeeze(polyappro), key=lambda x: -x[1], reverse=True)
                if len(sorted_poly) >= 2:
                    topleft = min(sorted_poly, key=lambda x: 0.3 * x[0] + 0.7 * x[1])
                    botleft = min(sorted_poly, key=lambda x: x[0] - x[1])
                    botright = max(sorted_poly, key=lambda x: x[0] + x[1])
                    botpos = (topleft[0] - botleft[0]) / (topleft[1] - botleft[1]) * (480 - topleft[1]) + topleft[0]
                    botpos = int(botpos)
                    botposes.append(botpos)

                    botlefts[0].append(botleft[0])
                    botlefts[1].append(botleft[1])
                    botrights[0].append(botright[0])
                    botrights[1].append(botright[1])
                    angle = utils.getangle(botleft, botright)
                    angles.append(angle)
                else:
                    print('拟合多边形边数小于2')
            time.sleep(0.05)  # 等待获取下一张图片
    # 取中位数，确保鲁棒性
    if len(angles):
        botleft = [int(statistics.median(botlefts[0])), int(statistics.median(botlefts[1]))]
        botright = [int(statistics.median(botrights[0])), int(statistics.median(botrights[1]))]
        botpos = int(statistics.median(botposes))
        angle = statistics.median(angles)
        return angle, botleft, botright, botpos


def findlow_door(app_e, threshold, key=cv2.contourArea):
    """
    胸部摄像头，找指定key下最大轮廓的最低边线
    :param app_e: 多边形近似程度
    :param threshold: 掩膜范围
    :param key: 比较函数，用于筛选符合条件的轮廓
    :return: 底线角度和端点
    """
    angles = []
    loilefts = [[], []]
    loirights = [[], []]

    def compare(points):
        mediumy = (points[0][1] + points[1][1]) / 2
        len = math.sqrt((points[0][1] - points[1][1]) ** 2 + (points[0][0] - points[1][0]) ** 2)
        angle = abs(utils.getangle(points[0], points[1]))
        comp = 0.6 * mediumy + 0.1 * len + 0.3 * (-angle)
        return comp

    for _ in range(5):
        if ChestOrg_img is not None:
            img_cop = ChestOrg_img.copy()
            img_cop = cv2.resize(img_cop, (640, 480))
            hsv = cv2.cvtColor(img_cop, cv2.COLOR_BGR2HSV)
            Imask = cv2.inRange(hsv, threshold[0], threshold[1])
            Imask = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, np.ones((7, 7)), iterations=3)
            Imask = cv2.morphologyEx(Imask, cv2.MORPH_CLOSE, np.ones((5, 5)), iterations=2)
            if Debug:
                cv2.imwrite('mask.jpg', Imask)

            # 指定轮廓最低边线
            contours, _ = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_contour = max(contours, key=key)
            poly = cv2.approxPolyDP(max_contour, app_e * cv2.arcLength(max_contour, True), True)
            line = []
            for i in range(len(poly)):
                line.append((np.squeeze(poly[i - 1]), np.squeeze(poly[i])))
            line = list(filter(lambda x: abs(utils.getangle(x[0], x[1])) < 40, line))
            line = sorted(line, key=compare, reverse=True)

            loi = line[0]
            angle = utils.getangle(loi[0], loi[1])
            loilefts[0].append(loi[0][0])
            loilefts[1].append(loi[0][1])
            loirights[0].append(loi[1][0])
            loirights[1].append(loi[1][1])
            angles.append(angle)
        time.sleep(0.05)  # 等待获取下一张图片
    # 取中位数，确保鲁棒性
    if len(angles):
        loileft = [int(statistics.median(loilefts[0])), int(statistics.median(loilefts[1]))]
        loiright = [int(statistics.median(loirights[0])), int(statistics.median(loirights[1]))]
        if loileft[0] > loiright[0]:
            temp = loiright
            loiright = loileft
            loileft = temp
        angle = statistics.median(angles)
        return angle, loileft, loiright


def imgpre(offset=0, Chest=True, Chest_thre=bluedoor_color_range['blue_door_chest'],
           Head_thre=bluedoor_color_range['green_bridge']):
    """
    图像处理函数
    :param offset: 角度偏置
    :param Chest: 胸部摄像头标志位
    :param Chest_thre: 胸部处理所需hsv范围
    :param Head_thre: 头部处理所需hsv范围
    :return: 底线端点，中点位置，角度
    """
    global Debug
    if ChestOrg_img is not None and Chest:
        loi = []
        img_cop = HeadOrg_img.copy()
        img_cop = cv2.resize(img_cop, (640, 480))
        angle, loileft, loiright = findlow_door(0.01, Chest_thre)
        loi.append(loileft)
        loi.append(loiright)
        angle = angle - offset
        pos = (loileft[1] + loiright[1]) / 2
        return loi, pos, angle
    elif HeadOrg_img is not None:
        bot = []
        img_cop = HeadOrg_img.copy()
        img_cop = cv2.resize(img_cop, (640, 480))
        angle, botleft, botright, botpos = get_robust_angle(0.01, Head_thre)
        bot.append(botleft)
        bot.append(botright)
        angle = angle + offset
        pos = (botleft[0] + botright[0]) / 2
        return bot, pos, angle, botpos


def door():
    global state, state_sel, step, angle_change, Debug
    state_sel = 'door'
    cnt_you = 0
    shif_bot = 0
    loi_bef = None  # 上一个loi，胸部摄像底线
    bot_bef = None  # 上一个bot，头部摄像底线
    if state_sel == 'door':  # 初始化
        print("开始门")
        step = 0
    while state_sel == 'door':
        if HeadOrg_img is not None and ChestOrg_img is not None:
            if step == 0:  # 姿态预调整
                print('##################step=0####################\n 姿态预调整')
                print('后退，右转')
                for _ in range(2):
                    print('向后退')
                    utils.act('Backward0')
                for _ in range(5):
                    print('向右转')
                    utils.act('turnR2')
                step = 1

            elif step == 1:  # 面对挡板调整角度
                over_pos = False
                print('#################step=1####################')
                # 图像预处理
                loi, pos, angle = imgpre(offset=angle_set['step1'][1])

                ############
                if Debug:
                    img_cop = ChestOrg_img.copy()
                    print('左端点', loi[0][0], '  中点y值', pos, ' jiaodu', angle)
                    imgstep1 = cv2.line(img_cop, tuple(loi[0]), tuple(loi[1]), (0, 0, 255), 2)
                    imgstep1 = cv2.circle(imgstep1, tuple(loi[0]), 3, (0, 255, 0), -1)
                    # cv2.putText(imgstep1,str(f'端点{loi}中心{pos}'),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255))
                    cv2.imwrite('step1.jpg', imgstep1)
                # 动作执行
                if loi[0][0] > pos_set['step1'][4] and abs(angle)<3:
                    print('先右移一下')
                    utils.act('panR1')
                if pos > pos_set['step1'][3] and abs(angle)<3:
                    print('先后退一下')
                    utils.act('Backward0')
                    over_pos = True
                elif pos < pos_set['step1'][2] and abs(angle)<3:
                    print('先前进一下')
                    utils.act('Forward0')
                    over_pos = True

                if angle > angle_set['step1'][0]:
                    if angle > angle_set['step1'][0] + 2:
                        print('大左转')
                        utils.act('turnL1')
                        continue
                    print('小左转')
                    utils.act('turnL0')
                elif angle < -angle_set['step1'][0]:
                    if angle < -angle_set['step1'][0] - 2:
                        print('大右转')
                        utils.act('turnR1')
                        continue
                    print('小右转')
                    utils.act('turnR0')
                elif not over_pos:
                    print('对准墙了')
                    if pos > pos_set['step1'][1]:
                        print('后退')
                        utils.act('Backward0')
                    elif pos < pos_set['step1'][0]:
                        print('前进')
                        utils.act('Forward0')
                    else:
                        print('位置合适')
                        print('转头')
                        utils.act('HeadturnR')
                        step = 2

            elif step == 2:  # 都用头对正
                print('#################step=2####################')
                bot, pos_x, angle, botpos = imgpre(offset=angle_set['step2'][angle_change], Chest=False)
                pos_y = (bot[0][1] + bot[1][1]) / 2
                loi, _, _ = imgpre()

                ############
                if Debug:
                    img_cop = HeadOrg_img.copy()
                    print('连接后底线交点', botpos, '  底线中点', (pos_x, pos_y), '  角度', angle, '  loi', loi)
                    imgstep2 = cv2.circle(img_cop, tuple([int(pos_x), int(pos_y)]), 3, (0, 255, 0), -1)
                    imgstep2 = cv2.circle(imgstep2, tuple([int(botpos), 480]), 3, (0, 255, 0), -1)
                    cv2.line(imgstep2, tuple(bot[0]), tuple(bot[1]), (0, 0, 255), 2)
                    cv2.imwrite('imgstep2.jpg', imgstep2)
                if cnt_you>=6:
                    print('进入胸部摄像头识别阶段')
                    step=3

                if cnt_you >= 4:
                    angle_change = 2
                    pos_set['step2'][0] += pos_change_min
                    pos_set['step2'][2] += pos_change_min
                    pos_set['step2'][1] += pos_change_max
                    pos_set['step2'][3] += pos_change_max
                # 动作执行

                if bot_bef is not None:
                    if abs(bot_bef - botpos) <= 5 and shif_bot < 10:
                        shif_bot += bot_bef - botpos
                        print('位置合适继续右移')
                        utils.act('panR1_d')
                        utils.act('panR1_d')
                        utils.act('panR1_d')
                        cnt_you += 3
                        continue

                if pos_x > pos_set['step2'][3] and abs(angle) < angle_set['step2'][0]:
                    print('先后退一下')
                    utils.act('Backward0_d')
                    continue
                elif pos_x < pos_set['step2'][2] and abs(angle) < angle_set['step2'][0]:
                    print('先前进一下')
                    utils.act('Forward0_d')
                    continue

                if angle > angle_set['step2'][0]:
                    print('需要左转')
                    utils.act('turnL0_d')
                elif angle < -angle_set['step2'][0]:
                    print('需要右转')
                    utils.act('turnR0_d')
                else:
                    print('站正了')
                    if pos_x > pos_set['step2'][1]:
                        print('后退')
                        utils.act('Backward0_d')
                    elif pos_x < pos_set['step2'][0]:
                        print('前进')
                        utils.act('Forward0_d')
                    else:
                        print('位置合适')
                        if loi[1][0] < pos_set['step2'][4]:
                            loi_bef = loi
                            bot_bef = botpos
                            print('向右侧前进')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            cnt_you += 6
                        else:
                            print('向右侧前进')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            utils.act('panR1_d')
                            cnt_you += 6
                            if (loi_bef is not None and loi[1][0] - loi_bef[1][0] > pos_set['step2'][5]) \
                                    or pos_y > pos_set['step2'][6]:
                                print('进入胸部摄像头识别阶段')
                                step = 3

            elif step == 3:  # 都用胸对正
                print('step=3#####################')
                loi, pos, angle = imgpre(offset=angle_set['step3'][1])

                ##########
                if Debug:
                    img_cop = ChestOrg_img.copy()
                    print('y值', pos)
                    print(loi, '  ', angle)
                    imgstep3 = cv2.line(img_cop, tuple(loi[0]), tuple(loi[1]), (0, 255, 0), 2)
                    # cv2.putText(imgstep3,str(f'端点{loi}中心{pos}'),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,255))
                    cv2.imwrite('step3.jpg', imgstep3)
                # 动作执行
                if pos < pos_set['step3'][2] and abs(angle) < angle_set['step3'][0]:
                    print('先前近一下')
                    utils.act('Forward0')
                    continue
                elif pos > pos_set['step3'][3] and abs(angle) < angle_set['step3'][0]:
                    print('先后退一下')
                    utils.act('Backward0')
                    continue

                if loi_bef is not None:
                    if (loi[1][0] + loi_bef[1][0]) / 2 < pos_set['step3'][4]:
                        print('即将通关，向右走两步，向右转弯90度')
                        for _ in range(2):
                            utils.act('panR1')
                        for _ in range(3):
                            utils.act('turnR2')
                        break

                if angle > angle_set['step3'][0]:
                    print('需要左转')
                    utils.act('turnL0')
                elif angle < -angle_set['step3'][0]:
                    print('需要右转')
                    utils.act('turnR0')
                else:
                    print('站正了')
                    if pos > pos_set['step3'][0]:
                        print('后退')
                        utils.act('Backward0')
                    elif pos < pos_set['step3'][1]:
                        print('前进')
                        utils.act('Forward0')
                    else:
                        print('距离合适,向右移')
                        utils.act('panR1')
                        utils.act('panR1')
                        utils.act('panR1')
                        loi_bef = loi
                        continue


########################################################################
##################             过独木桥              ####################
########################################################################


def get_robust_angle_bridge(app_e, threshold):
    """
    获取远处或近处底线角度
    :param app_e: 拟合多边形程度
    :param threshold 颜色阈值
    :return: 角度值，正值应左转，负值应右转
    """
    angles = []
    # 获取多张照片
    for _ in range(5):
        if ChestOrg_img is not None:
            img = ChestOrg_img.copy()
            img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, threshold[0], threshold[1])

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 5000:
                polyappro = cv2.approxPolyDP(
                    max_cnt, epsilon=app_e*cv2.arcLength(max_cnt, closed=True), closed=True)
                sorted_poly = sorted(np.squeeze(polyappro),
                                     key=lambda x: -x[1], reverse=True)
                if len(sorted_poly) >= 2:
                    topleft = max(sorted_poly, key=lambda x: -x[0]-x[1])
                    topright = max(sorted_poly, key=lambda x: x[0]-x[1])
                    bottomleft = max(sorted_poly, key=lambda x: -x[0]+x[1])
                    bottomright = max(sorted_poly, key=lambda x: x[0]+x[1])

                    if chest_width-bottomleft[1] < 50:  # 用远端线判断
                        left = topleft
                        right = topright
                    else:
                        left = bottomleft
                        right = bottomright
                    angle = utils.getangle(left, right)
                    angles.append(angle)
                    if Debug:
                        debug = cv2.line(img.copy(), tuple(left), tuple(
                            right), (0, 0, 255), thickness=3)
                        debug = cv2.circle(debug, tuple(
                            left), 3, (255, 0, 0), -1)
                        green = cv2.bitwise_and(
                            img.copy(), img.copy(), mask=mask)
                        cv2.imwrite('./log/bridge/'+utils.getlogtime() +
                                    'angle_correction_line.jpg', debug)
                        cv2.imwrite('./log/bridge/'+utils.getlogtime() +
                                    'angle_correction_green.jpg', green)
                else:
                    print('拟合多边形边数小于2')

            time.sleep(0.05)  # 等待获取下一张图片
    # 取中位数，确保鲁棒性
    if len(angles):
        angle = statistics.median(angles)
        return angle


def get_horizonal_position_bridge(app_e, frame_middle_x, threshold):
    if ChestOrg_img is not None:
        img = ChestOrg_img.copy()
        img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, threshold[0], threshold[1])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            contours = filter(lambda x: cv2.contourArea(x) > 1000, contours)
            max_cnt = max(contours, key=cv2.contourArea)
            polyappro = cv2.approxPolyDP(
                max_cnt, epsilon=app_e*cv2.arcLength(max_cnt, closed=True), closed=True)
            polyappro = np.squeeze(polyappro)
            leftbottom = max(polyappro, key=lambda x: -x[0]+x[1])
            lefttop = max(polyappro, key=lambda x: -x[0]-x[1])

            # 求leftbottom和lefttop连线和底部的交点
            y = chest_width
            x = ((y-leftbottom[1])*lefttop[0]-(y-lefttop[1])
                 * leftbottom[0])/(lefttop[1]-leftbottom[1])

            dist = frame_middle_x - x

            x = int(x)
            y = int(y)
            if Debug:
                debug = cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                cv2.imwrite('./log/bridge'+utils.getlogtime()+'.jpg', debug)
            return dist


def cross_narrow_bridge():
    print('进入过窄桥关')
    threshold = bridge_color_range
    """
    过窄桥
    """
    print("###############################################")
    orintation_right = False
    horizonal_right = False
    while True:
        if ChestOrg_img is None:
            print('相机未准备好')
            time.sleep(1)
            continue
        area = find_max_area(threshold)
        angle = get_robust_angle_bridge(0.01, threshold)
        print('当前绿色面积：', area)
        if area is None or area < 5000:
            print('往前走通过本关')
            utils.act('Forward1')
            break

        if angle is None:
            print('没找到角度，后退一点')
            utils.act('Backward0')
            continue

        print('角度：', angle)
        if -3 < angle < 3:
            print('朝向正确')
            orintation_right = True
        elif angle <= -3:
            orintation_right = False
            print('右转')
            utils.act('turnR0')
        elif angle >= 3:
            orintation_right = False
            print('小左转', angle)
            utils.act('turnL0')

        if orintation_right:  # 朝向正确，检查左右偏移
            pos = get_horizonal_position_bridge(0.01, 320, threshold)
            print('左边边界位置:', pos)
            if 115 < pos < 195:  # 待修改
                horizonal_right = True
            if pos <= 115:
                horizonal_right = False
                print('小右移')
                utils.act('panR0_1')
            if pos >= 195:
                horizonal_right = False
                print('小左移')
                utils.act('panL0')

        if orintation_right and horizonal_right:
            print('向前走')
            utils.act('Forward1')


def getParameters_bridge():
    threshold = bridge_color_range
    angle = get_robust_angle_bridge(0.01, threshold)
    pos = get_horizonal_position_bridge(0.01, 320, threshold)
    area = find_max_area(threshold)

    print('角度', angle)
    print('位置', pos)
    print('面积', area)

########################################################################
##################               踢球               #####################
########################################################################


# 以下是需要调整的参数（胸部摄像机颜色阈值）
#######################################################
ball_color_range = {'brick': [(139, 32, 68), (179, 104, 145)],
                    'ball': [(0, 0, 141), (134, 187, 255)],
                    'blue': [(92,111,82),(134,238,219)]}
#######################################################


def find_track_mask(img):
    """
    寻找赛道掩模
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_brick = cv2.inRange(hsv, ball_color_range['brick'][0], ball_color_range['brick'][1])
    mask_blue = cv2.inRange(hsv, ball_color_range['blue'][0], ball_color_range['blue'][1])
    # 防止左侧楼梯干扰
    mask_blue[:200, :200
              ] = 0

    mask_track = cv2.bitwise_or(mask_brick, mask_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_track = cv2.morphologyEx(
        mask_track, cv2.MORPH_CLOSE, kernel, iterations=5)

    contours, _ = cv2.findContours(
        mask_track, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    poly = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
    mask_track = np.zeros_like(mask_track)
    cv2.drawContours(mask_track, [poly], -1, 255, -1)

    # cv2.imshow('trr',cv2.bitwise_and(img,img,mask=mask_track))

    return mask_track, poly


def find_ball(img, mask_track):
    """
    寻找球心
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ball_color_range['brick'][0], ball_color_range['brick'][1])
    # 球裁剪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 对赛道的下半部分使用较大的闭运算
    mask_down = mask.copy()
    mask_down = cv2.morphologyEx(
        mask_down, cv2.MORPH_CLOSE, kernel, iterations=15)  # 闭运算封闭连接
    # 对赛道的上半部分使用较小的闭运算
    mask_up = mask.copy()
    mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 对赛道的中间部分使用适中的闭运算
    mask_med = mask.copy()
    mask_med = cv2.morphologyEx(
        mask_med, cv2.MORPH_CLOSE, kernel, iterations=9)
    # cv2.imshow('up', cv2.bitwise_and(img, img, mask=mask_up))
    # cv2.imshow('down', cv2.bitwise_and(img, img, mask=mask_down))
    mask_ball1 = np.zeros_like(mask)
    mask_ball1[:240, :] = mask_up[:240, :]
    mask_ball1[240:350, :] = mask_med[240:350, :]
    mask_ball1[350:, :] = mask_down[350:, :]
    mask_ball1 = cv2.bitwise_not(mask_ball1)
    mask_ball1[460:, :] = 0
    mask_ball2 = cv2.inRange(hsv, ball_color_range['ball'][0], ball_color_range['ball'][1])
    mask_ball = cv2.bitwise_and(mask_ball1, mask_ball2)
    mask_ball = cv2.bitwise_and(mask_ball, mask_track)
    mask_ball = cv2.morphologyEx(
        mask_ball, cv2.MORPH_OPEN, kernel, iterations=4)
    mask_ball = cv2.morphologyEx(
        mask_ball, cv2.MORPH_CLOSE, kernel, iterations=1)

    #
    # cv2.imshow('ball', cv2.bitwise_and(img, img, mask=mask_ball))

    # 边缘检测
    edges = cv2.Canny(mask_ball, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    contours = filter(lambda x: cv2.contourArea(x) > 10, contours)

    ratio = np.inf
    area = 0
    y = 480
    target_cnt = None
    # 找出长宽比最接近1的轮廓
    # 二次修正：添加面积权重
    # 三次修正：添加y值权重
    for cnt in contours:
        _, edge_len, _ = cv2.minAreaRect(cnt)
        ratio_tmp = math.fabs(math.log(edge_len[0]/(edge_len[1]+1e-6)))
        area_tmp = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        y_tmp = M['m01']/(M['m00']+1e-6)
        if 0.1*y_tmp+ratio_tmp-0.001*area_tmp < 0.1*y+ratio-0.001*area:
            ratio = ratio_tmp
            area = area_tmp
            y = y_tmp
            target_cnt = cnt
    # 根据不同情况计算中心和半径
    area = cv2.contourArea(target_cnt)
    if area > 500:
        M = cv2.moments(target_cnt)
        center_x = int(M['m10']/(M['m00']+1e-6))
        center_y = int(M['m01']/(M['m00']+1e-6))
        center = (center_x, center_y)

        circ = cv2.arcLength(target_cnt, True)

        r = int(area/circ*2)
    else:
        center, r = cv2.minEnclosingCircle(target_cnt)
        center = (int(center[0]), int(center[1]))
        r = int(r)

    cv2.drawContours(img, [target_cnt], -1, (255, 0, 0), 1)
    img = cv2.circle(img, center, r, (0, 0, 255), 1)
    # cv2.imshow('ballline', img)

    return r, center[0], center[1]


def find_hole(img, track_mask):
    """
    寻找球洞
    """

    # 将赛道掩膜下半部分置零，防止脚部影响
    track_mask[240:, :] = 0

    trackimg = cv2.bitwise_and(img, img, mask=track_mask)
    trackimg_hsv = cv2.cvtColor(trackimg, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(trackimg_hsv, ball_color_range['blue'][0], ball_color_range['blue'][1])

    contours, _ = cv2.findContours(
        mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(cnt)
    x = M['m10']/(M['m00']+1e-6)
    y = M['m01']/(M['m00']+1e-6)

    x = int(x)
    y = int(y)

    return x, y


def find_remote_edge(polydp):
    """
    寻找远端赛道边缘
    """
    def norm(points):
        return (points[0][0]-points[1][0])**2+(points[0][1]-points[1][1])**2

    def medium_y(points):
        return (points[0][1]+points[1][1])/2

    polydp = np.squeeze(polydp)
    edges = []
    for i in range(len(polydp)):
        edges.append((polydp[i-1], polydp[i]))
    edges_filtered = filter(lambda x: norm(x) > 10000, edges)

    selected_edge = min(edges_filtered, key=medium_y)

    return selected_edge[0], selected_edge[1]


def kickball():
    class Step(Enum):
        WALK2BALL = 1
        ADJUST2KICK = 2
        KICK = 3
        FINISHKICK = 4

    print('进入踢球关')
    step = Step.WALK2BALL

    while True:
        if ChestOrg_img is None or HeadOrg_img is None:
            print('摄像头未准备好')
            time.sleep(1)
            continue

        chestimg = ChestOrg_img.copy()
        headimg = HeadOrg_img.copy()

        # 通过侧移和前进的方式靠近球
        if step == Step.WALK2BALL:
            # 以下是需要调整的参数
            ################################################################################
            angle_threshold = (-3, 3)  # 机器人角度
            ball_center_threshold = (320, 340)  # 让球中心保持在这个位置之间
            distance_threshold = 100  # 球心距小于这个值时进入下一个阶段
            area_threshold = 10000  # 球面积超过这个值时进入下一个阶段
            ################################################################################

            # 获取各项数据
            track_mask, poly = find_track_mask(chestimg)
            r_ball, x_ball, y_ball = find_ball(chestimg, track_mask)
            left, right = find_remote_edge(poly)

            # 计算角度
            angle = utils.getangle(left, right)
            print('当前朝向角', angle)

            orintation_right = False
            position_right = False

            # 调整转向
            if angle_threshold[0] < angle < angle_threshold[1]:
                orintation_right = True
                print('朝向正确')
            elif angle <= angle_threshold[0]:
                orintation_right = False
                print('需要右转')
                utils.act('turnRight')
            elif angle >= angle_threshold[1]:
                orintation_right = False
                print('需要左转')
                utils.act('turnLeft')

            # 左右调整位置
            if orintation_right:
                print('当前球心x值', x_ball)
                if ball_center_threshold[0] < x_ball < ball_center_threshold[1]:
                    position_right = True
                    print('位置正确')
                elif x_ball <= ball_center_threshold[0]:
                    position_right = False
                    print('需要左移')
                    utils.act('leftshift')
                elif x_ball > ball_center_threshold[1]:
                    position_right = False
                    print('需要右移')
                    utils.act('rightshift')

            # 判断球距
            if orintation_right and position_right:
                area = math.pi*r_ball**2
                dist = chest_width-y_ball

                if dist < distance_threshold and area > area_threshold:
                    print(
                        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n进入调整位置阶段！\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    step = Step.ADJUST2KICK

        # 调整身位踢球
        if step == Step.ADJUST2KICK:
            # 以下是需要调整的参数
            #########################################################################
            verticle_threshold = 80  # 球和洞连线斜角阈值
            ball_center_threshold = (320, 340)  # 让球中心保持在这个位置之间
            #########################################################################

            # 获取各项数据
            track_mask, poly = find_track_mask(chestimg)
            r_ball, x_ball, y_ball = find_ball(chestimg, track_mask)
            x_hole, y_hole = find_hole(chestimg, track_mask)

            left = (x_ball, y_ball)
            right = (x_hole, y_hole)
            angle = utils.getangle(left, right)

            orintation_ready = False
            position_ready = False

            # 调整角度
            print('当前球洞角', angle)
            if math.fabs(angle) > verticle_threshold:
                orintation_ready = True
                print('球洞线垂直')
            elif angle >= 0:
                orintation_ready = False
                print('需要右转')
                utils.act('turnRight')
            elif angle < 0:
                orintation_ready = False
                print('需要左转')
                utils.act('turnLeft')

            # 调整位置
            # 位置的判定用球洞的连线与相机底边框的交点
            y = chest_width
            x = ((y-y_hole)*x_ball-(y-y_ball)*x_hole)/(y_ball-y_hole)

            if ball_center_threshold[0] < x < ball_center_threshold[1]:
                position_ready = True
                print('位置正确')
            elif x <= ball_center_threshold[0]:
                position_ready = False
                print('需要左移')
                utils.act('shiftleft')
            elif x > ball_center_threshold[1]:
                position_ready = False
                print('需要右移')
                utils.act('shiftright')
            if Debug:
                line = cv2.line(chestimg, (x_hole, y_hole),
                                (x, y), (0, 0, 255), 2)
                cv2.imwrite('./log/ball/'+utils.getlogtime() +
                            'hole_ball.jpg', line)
            if orintation_ready and position_ready:
                print(
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n进入踢球阶段！\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                step = Step.KICK

        if step == Step.KICK:
            # 以下是需要调整的参数
            #######################################################################
            distance_threshold = 50  # 踢球时球心的位置
            #######################################################################

            # 获取各项数据
            track_mask, poly = find_track_mask(chestimg)
            r_ball, x_ball, y_ball = find_ball(chestimg, track_mask)

            dist = chest_height-y_ball

            if dist > distance_threshold:
                print('向前走一小步')
                utils.act('smallforward')
            else:
                print(
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n踢球！\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                utils.act('fastforward')
                print('踢球结束，进入下一关')

        if step == Step.FINISHKICK:
            pass

# 调试参数


def getParameters_ball():

    if ChestOrg_img is not None:

        chestimg = ChestOrg_img.copy()

        track_mask, poly = find_track_mask(chestimg)
        r_ball, x_ball, y_ball = find_ball(chestimg, track_mask)
        x_hole, y_hole = find_hole(chestimg, track_mask)
        left, right = find_remote_edge(poly)

        y = chest_width
        x = ((y-y_hole)*x_ball-(y-y_ball)*x_hole)/(y_ball-y_hole)

        angle1 = utils.getangle(left, right)

        img = cv2.line(chestimg, tuple(left),tuple(right), (0, 0, 255), 2)

        left = (x_ball, y_ball)
        right = (x_hole, y_hole)

        angle2 = utils.getangle(left, right)

        img = cv2.line(img, left, right, (255, 0, 0), 2)
        area = math.pi*r_ball**2

        dist = chest_width - y_ball

        cv2.imwrite('./log/ball/'+utils.getlogtime()+'ballinfo.jpg',img)
        print('###################################')
        print('底边线角度:', angle1)
        print('球心x坐标:', x_ball)
        print('球洞延长线交点:', x)
        print('球心距离:', dist)
        print('白球面积:', area)
        print('球洞角:', angle2)

###########################################################################
##########                       楼梯                             ##########
###########################################################################

def floor():
    global org_img, state, state_sel, step
    state_sel = 'floor'
    img_debug = 0
    if state_sel == 'floor':  # 初始化
        print("/-/-/-/-/-/-/-/-/-进入floor")
        step = 0

    r_w = chest_width
    r_h = chest_height

    top_angle = 0
    T_B_angle = 0
    topcenter_x = 0.5 * r_w
    topcenter_y = 0
    bottomcenter_x = 0.5 * r_w
    bottomcenter_y = 0

    while state_sel == 'floor':

        # 分析图像
        # chest
        if True:  # 上下边沿
            OrgFrame = ChestOrg_img.copy()
            # OrgFrame=cv2.imread('D:desktop/stairforward.jpg')

            # 初始化 bottom_right  bottom_left
            bottom_right = (480, 0)
            bottom_left = (0, 0)
            top_right = (480, 0)  # 右上角点坐标
            top_left = (0, 0)  # 左上角点坐标

            frame = OrgFrame
            # frame_copy = frame.copy()########################
            frame_copy = frame
            # 获取图像中心点坐标x, y
            # 开始处理图像
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #####################

            lab = cv2.GaussianBlur(lab, (7, 7), 0)  # 高斯滤波
            if step == 0:
                Imask = cv2.inRange(lab, stair_color_range['blue_floor'][0],
                                    stair_color_range['blue_floor'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            elif step == 1:
                Imask = cv2.inRange(lab, stair_color_range['blue_floor'][0], stair_color_range['blue_floor'][1])
            elif step == 2:
                Imask = cv2.inRange(lab, stair_color_range['green_floor'][0], stair_color_range['green_floor'][1])
            elif step == 3:
                Imask = cv2.inRange(lab, stair_color_range['red_floor'][0], stair_color_range['red_floor'][1])
            elif step == 4:
                Imask = cv2.inRange(lab, stair_color_range['green_floor'][0], stair_color_range['green_floor'][1])
            elif step == 5:
                Imask = cv2.inRange(lab, stair_color_range['blue_floor'][0], stair_color_range['blue_floor'][1])
            elif step == 6:
                Imask = cv2.inRange(lab, stair_color_range['red_floor'][0],
                                    stair_color_range['red_floor'][1])
            elif step == 7:
                Imask = cv2.inRange(lab, stair_color_range['blue_floor'][0],
                                    stair_color_range['blue_floor'][1])  # 取决于后面的关卡
            else:
                print("no color")
                open = frame_copy

            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(Imask, cv2.MORPH_CLOSE, kernal)
            open = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernal)
            cnts, hierarchy = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
            cnt_sum, area_max = utils.getAreaMaxContour1(cnts)  # 找出最大轮廓
            cv2.drawContours(OrgFrame, cnts, -1, (255, 0, 255), 1)
            C_percent = round(100 * area_max / (chest_width * chest_height), 2)  # 最大轮廓1的百分比

            # cv2.drawContours(frame, cnt_sum, -1, (255, 0, 255), 3)
            if cnt_sum is not None:
                bottom_right = cnt_sum[0][0]  # 右下角点坐标
                bottom_left = cnt_sum[0][0]  # 左下角点坐标
                top_right = cnt_sum[0][0]  # 右上角点坐标
                top_left = cnt_sum[0][0]  # 左上角点坐标
                for c in cnt_sum:

                    if c[0][0] + 1 * (r_h - c[0][1]) < bottom_left[0] + 1 * (r_h - bottom_left[1]):
                        bottom_left = c[0]
                    if c[0][0] + 1 * c[0][1] > bottom_right[0] + 1 * bottom_right[1]:
                        bottom_right = c[0]

                    if c[0][0] + 3 * c[0][1] < top_left[0] + 3 * top_left[1]:
                        top_left = c[0]
                    if (r_w - c[0][0]) + 3 * c[0][1] < (r_w - top_right[0]) + 3 * top_right[1]:
                        top_right = c[0]

                bottomcenter_x = (bottom_left[0] + bottom_right[0]) / 2  # 得到bottom中心坐标
                bottomcenter_y = (bottom_left[1] + bottom_right[1]) / 2
                topcenter_x = (top_right[0] + top_left[0]) / 2  # 得到top中心坐标
                topcenter_y = (top_left[1] + top_right[1]) / 2

                bottom_angle = -math.atan(
                    (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0] + 1e-4)) * 180.0 / math.pi
                top_angle = -math.atan(
                    (top_right[1] - top_left[1]) / (top_right[0] - top_left[0] + 1e-4)) * 180.0 / math.pi
                if math.fabs(topcenter_x - bottomcenter_x) <= 1:  # 得到连线的角度
                    T_B_angle = 90
                else:
                    T_B_angle = - math.atan(
                        (topcenter_y - bottomcenter_y) / (topcenter_x - bottomcenter_x + 1e-4)) * 180.0 / math.pi

                if img_debug:
                    # cv2.drawContours(frame_copy, [box], 0, (0, 255, 0), 2)  # 将大矩形画在图上
                    cv2.line(frame_copy, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1]),
                             (255, 255, 0), thickness=2)
                    cv2.line(frame_copy, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (255, 255, 0),
                             thickness=2)
                    cv2.line(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)),
                             (int(topcenter_x), int(topcenter_y)), (255, 255, 255), thickness=2)  # T_B_line

                    cv2.putText(frame_copy, "bottom_angle:" + str(bottom_angle), (30, 450), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "top_angle:" + str(top_angle), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 2)
                    cv2.putText(frame_copy, "T_B_angle:" + str(T_B_angle), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 255), 2)

                    cv2.putText(frame_copy, "bottomcenter_x:" + str(bottomcenter_x), (30, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "bottomcenter_y:" + str(int(bottomcenter_y)), (300, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 0, 0), 2)  # (0, 0, 255)BGR

                    cv2.putText(frame_copy, "topcenter_x:" + str(topcenter_x), (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                    cv2.putText(frame_copy, "topcenter_y:" + str(int(topcenter_y)), (230, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR

                    cv2.putText(frame_copy, 'C_percent:' + str(C_percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)
                    cv2.putText(frame_copy, "step:" + str(step), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),
                                2)  # (0, 0, 255)BGR

                    cv2.circle(frame_copy, (int(topcenter_x), int(topcenter_y)), 5, [255, 0, 255], 2)
                    cv2.circle(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)), 5, [255, 0, 255], 2)
                    cv2.circle(frame_copy, (top_right[0], top_right[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (top_left[0], top_left[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (bottom_right[0], bottom_right[1]), 5, [0, 255, 255], 2)
                    cv2.circle(frame_copy, (bottom_left[0], bottom_left[1]), 5, [0, 255, 255], 2)
                    cv2.imshow('Chest_Camera', frame_copy)  # 显示图像
                    cv2.waitKey(1)
                    # cv2.imshow('chest_red_mask', Imask)

                # 决策执行动作
                if step == 0:
                    print('当前step = ', step)
                    if bottomcenter_y < 300:
                        if top_angle > 5:  # 需要左转
                            print("bottom_angle  需要小左转  ", top_angle)
                            utils.act("turnL0")
                        elif top_angle < -5:  # 需要右转
                            print("bottom_angle  需要小右转  ", top_angle)
                            utils.act("turnR0")
                        elif -5 <= top_angle <= 5:  # 角度正确
                            print("角度合适")

                            if topcenter_x < 280:  # look for?
                                print("微微左移,topcenter=", topcenter_x)
                                utils.act("panL0")
                            elif topcenter_x > 330:  # look for?
                                print("微微右移,topcenter=", topcenter_x)
                                utils.act("panR0")
                            else:
                                print("位置合适")
                                print("向前走,topcenter_x", topcenter_x)
                                print("向前走bottomcenter_y=", bottomcenter_y)
                                utils.act("Forward0")
                    elif 300 < bottomcenter_y < 430:  # look for?
                        if top_angle > 5:  # 需要左转
                            print("bottom_angle  需要小左转  ", top_angle)
                            utils.act("turnL0")
                        elif top_angle < -5:  # 需要右转
                            print("bottom_angle  需要小右转  ", top_angle)
                            utils.act("turnR0")
                        elif -5 <= top_angle <= 5:  # 角度正确
                            print("角度合适")
                            if topcenter_x < 280:  # look for?
                                print("微微左移,", topcenter_x)
                                utils.act("panL0")
                            elif topcenter_x > 330:  # look for?
                                print("微微右移,", topcenter_x)
                                utils.act("panR0")
                            else:
                                print("位置合适", topcenter_x)
                                print("向前走,bottomcenter_y", bottomcenter_y)
                                utils.act("Forward0")
                    elif bottomcenter_y > 430:  # look for ?
                        step = 1  # 进入第二步，上第一层楼梯
                        utils.act("Forward0")
                        print(' 进入第二步，上第一层楼梯')
                        print("bottomcenter_y:", bottomcenter_y)


                elif step == 1:
                    print('当前step = ', step)
                    if top_angle < -5:  # 右转
                        print("右转 top_angle:", top_angle)
                        utils.act("turnR0")
                    elif top_angle > 5:  # 左转
                        print("左转 top_angle:", top_angle)
                        utils.act("turnL0")
                    elif -5 <= top_angle <= 5:
                        if bottomcenter_y < 450:  #################
                            print("贴紧  bottomcenter_y=", bottomcenter_y)
                            utils.act("Forward0")
                            utils.act("Stand")
                            time.sleep(0.5)
                        else:
                            print(" 上台阶 buttomcenter=", bottomcenter_y)
                            utils.act("upstair")
                            print("开始上第二节台阶")
                            utils.act("Stand")
                            time.sleep(0.5)
                            step = 2

                elif step == 2:
                    print('当前step = ', step)
                    # if -88 < T_B_angle < 88:
                    #     if 0 < T_B_angle < 88:
                    #         print("微微左移,", T_B_angle)
                    #         utils.act("panL0")
                    #     elif -88 < T_B_angle < 0:
                    #         print("微微右移,", T_B_angle)
                    #         utils.act("panR0")
                    # else:
                    #     print("位置合适",T_B_angle)
                    #     if bottom_angle > 5:  # 需要左转
                    #         print("bottom_angle > 5 需要小左转 ", bottom_angle)
                    #         utils.act("turnL0")
                    #     elif bottom_angle < -5:  # 需要右转
                    #         print("bottom_angle < -5 需要小右转", bottom_angle)
                    #         utils.act("turnR0")
                    #     elif -5 <= bottom_angle <= 5:  # 角度正确
                    #         print("角度合适")
                    #         step = 3
                    # print("向前走,bottomcenter_y", bottomcenter_y)
                    # utils.act("Forward0")
                    print("上台阶 上台阶 upstair")
                    utils.act("upstair")
                    utils.act("Stand")
                    time.sleep(0.5)
                    # print('右移1大步')
                    # utils.act("panR1")
                    step = 3

                elif step == 3:
                    print('当前step = ', step)
                    # if bottom_angle > 5:  # 需要左转
                    #     print("bottom_angle > 5 需要小左转 ", bottom_angle)
                    #     utils.act("turnL0")
                    # elif bottom_angle < -5:  # 需要右转
                    #     print("bottom_angle < -5 需要小右转  ", bottom_angle)
                    #     utils.act("turnR0")
                    # elif -5 <= bottom_angle <= 5:  # 角度正确
                    #     print("角度合适",bottom_angle)
                    step = 4
                    print("4247L 上台阶 上台阶 upstair")
                    utils.act("upstair")
                    utils.act("Stand")
                    time.sleep(0.5)

                elif step == 4:
                    print('当前step = ', step)
                    if top_angle > 5:  # 需要左转
                        print("top_angle > 5 需要小左转 ", top_angle)
                        utils.act("turnL0")
                    elif top_angle < -5:  # 需要右转
                        print(" top_angle < -5 需要小右转 ", top_angle)
                        utils.act("turnR0")
                    elif -5 <= top_angle <= 5:  # 角度正确
                        print("角度合适")
                        if topcenter_x > 330:
                            print('需要右移 topcenter_x=', topcenter_x)
                            utils.act("panR0")
                        elif topcenter_x < 280:
                            print('需要左移 topcenter_x=', topcenter_x)
                            utils.act("panL0")
                        else:
                            print('位置合适 topcenter_x=', topcenter_x)

                            if bottomcenter_y < 485:  # look for?
                                print("微微前挪，bottomcenter_y=", bottomcenter_y)
                                utils.act("Forward0")
                            elif bottomcenter_y > 485:  # look for?
                                # print(" 下台阶 下台阶 DownBridge topcenter_y:", topcenter_y)
                                print("下阶梯no.1 bottomcenter_y=", bottomcenter_y)
                                utils.act("downstair")
                                utils.act("Stand")
                                time.sleep(0.5)
                                step = 5

                elif step == 5:
                    # print('当前step = ',step)
                    # if topcenter_y < 320:
                    #    print("微微前挪")
                    #    utils.act("Forward0")
                    # elif topcenter_y > 320:
                    # print("下台阶 下台阶 DownBridge topcenter_y:", topcenter_y)
                    print("下台阶no.2")
                    utils.act("downstair")
                    utils.act("Stand")
                    time.sleep(0.5)
                    step = 6

                elif step == 6:
                    if top_angle > 5:  # 需要左转
                        print("top_angle > 5 需要小左转 top_angle:", top_angle)
                        utils.act("turnL0")
                    elif top_angle < -5:  # 需要右转
                        print("top_angle < -5 需要小右转 top_angle:", top_angle)
                        utils.act("turnR0")
                    elif -5 <= top_angle <= 5:  # 角度正确
                        print("角度合适")
                        if topcenter_x > 330:
                            print('需要右移 topcenter_x', topcenter_x)
                            utils.act("panR1")
                        elif topcenter_x < 280:
                            print('需要左移 topcenter_x', topcenter_x)
                            utils.act("panL1")
                        else:
                            # if    310<topcenter_y<330:
                            print('位置合适 topcenter_x', topcenter_x)
                            print("下斜坡")
                            utils.act("Stand")
                            utils.act("Forward0")
                            utils.act("downslope1")
                            utils.act("downslope2")
                            time.sleep(0.5)
                            utils.act("downslope2")
                            time.sleep(0.5)
                            utils.act("downslope2")
                            time.sleep(0.5)
                            utils.act("downslope2")
                            utils.act("Stand")
                            utils.act("Forward0")
                            print("斜坡结束")
                            step = 7
                            # elif topcenter_y>350:######################
                            # print('调整位置 前进')
                            # utils.act("Forward0")
                            # else :
                            # print('调整位置 后退')
                            # utils.act("backward0")

                elif step == 7:
                    utils.act("Stand")
                    print("完成stair")
                    break
            else:
                print("未找到第一届蓝色台阶 左转")
                utils.act("turnL0")
                time.sleep(0.5)
    return


###########################################################################
##########                      终点门                            ##########
###########################################################################
def end_door():
    crossbardownalready = False
    intercept = [50, 430]
    PERCENT_THRESH = 0.05
    global ChestOrg_img
    while True:
        if goflag:
            utils.act("fastForward05")
            utils.act("Stand")
            print("成功通关！")

            # utils.act("forwardSlow0403")

            # utils.act("fast_forward_step")
            # cv2.destroyAllWindows()
            break


        else:  # 判断门是否打开
            handling = HeadOrg_img.copy()

            border = cv2.copyMakeBorder(handling, 12, 12, 16, 16, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))  # 扩展白边，防止边界无法识别
            handling = cv2.resize(border, (chest_width, chest_height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
            frame_gauss = cv2.GaussianBlur(handling, (21, 21), 0)  # 高斯模糊
            frame_hsv = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间

            frame_hsv = frame_hsv[intercept[0]:intercept[1], 0:640]  # 裁剪出图像要识别的部分

            frame_door_yellow = cv2.inRange(frame_hsv, end_door_color_range['yellow_door'][0],
                                            end_door_color_range['yellow_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            frame_door_black = cv2.inRange(frame_hsv, end_door_color_range['black_door'][0],
                                           end_door_color_range['black_door'][1])  # 对原图像和掩模(颜色的字典)进行位运算
            frame_door = cv2.add(frame_door_yellow, frame_door_black)

            open_pic = cv2.morphologyEx(frame_door, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 开运算 去噪点
            closed_pic = cv2.morphologyEx(open_pic, cv2.MORPH_CLOSE, np.ones((50, 50), np.uint8))  # 闭运算 封闭连接
            (contours, hierarchy) = cv2.findContours(closed_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓
            areaMaxContour, area_max = utils.getAreaMaxContour1(contours)  # 找出最大轮廓
            percent = round(100 * area_max / (chest_width * chest_height), 2)  # 最大轮廓的百分比

            if Debug:
                cv2.line(closed_pic, [0, intercept[0]], [640, intercept[0]], (100, 255, 100), 1)
                cv2.line(closed_pic, [0, intercept[1]], [640, intercept[1]], (100, 255, 100), 1)
                if percent > PERCENT_THRESH:
                    cv2.putText(closed_pic, percent, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(closed_pic, percent, (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imwrite('./closed_pic.jpg', closed_pic)  # 查看识别情况

            # 根据比例得到是否前进的信息
            if percent > PERCENT_THRESH:
                crossbardown = True
            else:
                crossbardown = False

            if not crossbardownalready:
                if crossbardown:
                    crossbardownalready = True
                    print("横杆已关闭，等待横杆开启")
                    print('percent = ', percent)
                else:
                    print("横杆未关闭，先等待横杆关闭")
                    print('percent = ', percent)
            else:
                if not crossbardown:
                    goflag = True
                    print("机器人启动")
                    utils.act('fastForward05')
                else:
                    print("横杆已关闭，等待横杆开启")
            time.sleep(0.1)
    return True

if __name__ == '__main__':
    rospy.init_node('runningrobot')
    while True:
        if ChestOrg_img is None or HeadOrg_img is None:
          time.sleep(1)
        else:
          break
    obstacle()