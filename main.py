#!/usr/bin/env python3
# coding:utf-8

import numpy as np
import cv2
import math
import threading
import time
import rospy
import sys
import utils
import statistics

chest_r_width = 480
chest_r_height = 640
head_r_width = 640
head_r_height = 480
ChestOrg_img = None  # 原始图像更新
HeadOrg_img = None  # 原始图像更新
r_width = 480
r_height = 640
Debug = True


#############更新图像#############
def updateImg():
    global ChestOrg_img, HeadOrg_img
    image_reader = utils.ImgConverter()
    while True:
        ChestOrg_img = image_reader.chest_image()
        HeadOrg_img = image_reader.head_image()
        time.sleep(0.05)

# 创建线程更新图像
th_capture = threading.Thread(target=utils.updateImg)
th_capture.setDaemon(True)
th_capture.start()

###########################################################################
##########                     过坑部分                           ###########
###########################################################################

hole_color_range = {
    'green_hole_chest': [(67, 60, 106), (90, 249, 255)],
    'blue_hole_chest': [(102, 123, 132), (110, 213, 235)],
}

def get_robust_angle(app_e,threshold):
    """
    获取远处底线角度
    :param app_e: 拟合多边形程度
    :param threshold 颜色阈值
    :return: 角度值，正值应左转，负值应右转
    """
    angles = []
    #获取多张照片
    for _ in range(10):
        if ChestOrg_img is not None:
            img = ChestOrg_img.copy()
            img = cv2.resize(img,(640,480),cv2.INTER_LINEAR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img,threshold[0],threshold[1])
            contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            coi = sorted(contours,key=cv2.contourArea,reverse=True)
            if len(coi):
                polyappro = cv2.approxPolyDP(coi[0],epsilon=app_e*cv2.arcLength(coi[0],closed=True),closed=True)
                sorted_poly = sorted(np.squeeze(polyappro),key=lambda x:-x[1],reverse=True)
                topleft = min(sorted_poly[:2],key=lambda x:x[0])
                topright = max(sorted_poly[:2],key=lambda x:x[0])
                angle = utils.getangle(topleft,topright)
                angles.append(angle)
            time.sleep(0.05)#等待获取下一张图片
    #取中位数，确保鲁棒性
    angle = statistics.median(angles)
    return angle

def adjust_orientation_hole(threshold):
    """
    根据底边线调整朝向
    :param threshold:地面颜色阈值
    """
    while True:
        angle = get_robust_angle(0.005,hole_color_range['green_hole_chest'])
        if -2 < angle < 2:
            print('朝向正确')
        elif angle <= -2:
            print('右转')
        elif angle >= 2:
            print('左转')
        time.sleep(1)


#识别过坑关卡
# 过坑识别
def hole_recognize(color):
    src = ChestOrg_img.copy()
    Area = 0
    src = src[int(100):int(400), int(50):int(500)]
    src = cv2.GaussianBlur(src, (5, 5), 0)
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hole_color_range[color][0], hole_color_range[color][1])
    closed = cv2.dilate(mask, None, iterations=5)
    closed = cv2.erode(closed, None, iterations=8)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        max_area = max(contours, key=cv2.contourArea)
        Area = cv2.contourArea(max_area)
        rect = cv2.minAreaRect(max_area)
        # print(rect[0])
        # # print(Area)
    contours2, hierarchy2 = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if Area > 20000 and len(contours2) >= 2:
        return True
    else:
        return False

# 根据颜色边缘调整角度与位置（胸部）
def edge_angle_chest(color):
    global ChestOrg_img, HeadOrg_img
    r_w = chest_r_width
    r_h = chest_r_height
    top_angle = 0
    T_B_angle = 0
    topcenter_x = 0.5 * r_w
    topcenter_y = 0
    bottomcenter_x = 0.5 * r_w
    bottomcenter_y = 0
    while (True):
        step = 0
        OrgFrame = ChestOrg_img.copy()

        # 初始化 bottom_right  bottom_left
        bottom_right = (480, 0)
        bottom_left = (0, 0)
        top_right = (480, 0)  # 右上角点坐标
        top_left = (0, 0)  # 左上角点坐标

        frame = cv2.resize(OrgFrame, (chest_r_width, chest_r_height), interpolation=cv2.INTER_LINEAR)
        # 获取图像中心点坐标x, y
        center = []
        # 开始处理图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask = cv2.inRange(hsv, hole_color_range[color][0], hole_color_range[color][1])
        Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=2)

        cnts, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓

        cnt_sum, area_max = utils.getAreaMaxContour(cnts)  # 找出最大轮廓
        C_percent = round(area_max * 100 / (r_w * r_h), 2)  # 最大轮廓百分比
        cv2.drawContours(frame, cnt_sum, -1, (255, 0, 255), 3)

        if cnt_sum is not None:
            see = True
            rect = cv2.minAreaRect(cnt_sum)  # 最小外接矩形
            box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点

            max_cnt = np.squeeze(cnt_sum)
            bottom_right = max(max_cnt, key=lambda x: x[0] + x[1])
            bottom_left = max(max_cnt, key=lambda x: -x[0] + x[1])
            top_right = max(max_cnt, key=lambda x: x[0] - x[1])
            top_left = max(max_cnt, key=lambda x: -x[0] - x[1])

            # bottom_right = cnt_sum[0][0]  # 右下角点坐标
            # bottom_left = cnt_sum[0][0]  # 左下角点坐标
            # top_right = cnt_sum[0][0]  # 右上角点坐标
            # top_left = cnt_sum[0][0]  # 左上角点坐标
            # for c in cnt_sum:
            #
            #     if c[0][0] + 1 * (r_h - c[0][1]) < bottom_left[0] + 1 * (r_h - bottom_left[1]):
            #         bottom_left = c[0]
            #     if c[0][0] + 1 * c[0][1] > bottom_right[0] + 1 * bottom_right[1]:
            #         bottom_right = c[0]
            #
            #     if c[0][0] + 3 * c[0][1] < top_left[0] + 3 * top_left[1]:
            #         top_left = c[0]
            #     if (r_w - c[0][0]) + 3 * c[0][1] < (r_w - top_right[0]) + 3 * top_right[1]:
            #         top_right = c[0]

            bottomcenter_x = (bottom_left[0] + bottom_right[0]) / 2  # 得到bottom中心坐标
            bottomcenter_y = (bottom_left[1] + bottom_right[1]) / 2

            topcenter_x = (top_right[0] + top_left[0]) / 2  # 得到top中心坐标
            topcenter_y = (top_left[1] + top_right[1]) / 2

            bottom_angle = -math.atan(
                (bottom_right[1] - bottom_left[1]) / (bottom_right[0] - bottom_left[0])) * 180.0 / math.pi
            top_angle = -math.atan((top_right[1] - top_left[1]) / (top_right[0] - top_left[0])) * 180.0 / math.pi
            if math.fabs(topcenter_x - bottomcenter_x) <= 1:  # 得到连线的角度
                T_B_angle = 90
            else:
                T_B_angle = - math.atan(
                    (topcenter_y - bottomcenter_y) / (topcenter_x - bottomcenter_x)) * 180.0 / math.pi

            if Debug:
                frame_copy = frame.copy()
                cv2.drawContours(frame_copy, [box], 0, (0, 255, 0), 2)  # 将大矩形画在图上
                cv2.line(frame_copy, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1]),
                         (255, 255, 0), thickness=2)
                cv2.line(frame_copy, (top_left[0], top_left[1]), (top_right[0], top_right[1]), (255, 255, 0),
                         thickness=2)
                cv2.line(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)), (int(topcenter_x), int(topcenter_y)),
                         (255, 255, 255), thickness=2)  # T_B_line

                cv2.putText(frame_copy, "bottom_angle:" + str(bottom_angle), (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "top_angle:" + str(top_angle), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2)
                cv2.putText(frame_copy, "T_B_angle:" + str(T_B_angle), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255), 2)

                cv2.putText(frame_copy, "bottomcenter_x:" + str(bottomcenter_x), (30, 480), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "y:" + str(int(bottomcenter_y)), (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2)  # (0, 0, 255)BGR

                cv2.putText(frame_copy, "topcenter_x:" + str(topcenter_x), (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2)  # (0, 0, 255)BGR
                cv2.putText(frame_copy, "topcenter_y:" + str(int(topcenter_y)), (230, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 2)  # (0, 0, 255)BGR

                cv2.putText(frame_copy, 'C_percent:' + str(C_percent) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2)
                cv2.putText(frame_copy, "step:" + str(step), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),
                            2)  # (0, 0, 255)BGR

                cv2.circle(frame_copy, (int(topcenter_x), int(topcenter_y)), 5, [255, 0, 255], 2)
                cv2.circle(frame_copy, (int(bottomcenter_x), int(bottomcenter_y)), 5, [255, 0, 255], 2)
                cv2.circle(frame_copy, (top_right[0], top_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (top_left[0], top_left[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (bottom_right[0], bottom_right[1]), 5, [0, 255, 255], 2)
                cv2.circle(frame_copy, (bottom_left[0], bottom_left[1]), 5, [0, 255, 255], 2)
                # cv2.imshow('Chest_Camera', frame_copy)  # 显示图像
                cv2.imwrite('./log/hole/'+utils.getlogtime()+'Chest_Camera.jpg', frame_copy)  # 查看识别情况

        else:
            print("815L  chest NONE")

        # 决策执行动作

        if step == 0:  # 前进依据chest 调整大致位置，方向  看底边线调整角度

            if top_angle > 2.5:  # 需要左转
                if top_angle > 6:
                    print("826L 大左转一下  turn001L ", top_angle)
                    utils.act("turn001L")
                else:
                    print("829L bottom_angle > 3 需要小左转 turn001L ", top_angle)
                    utils.act("turn001L")
            elif top_angle < -2.5:  # 需要右转
                if top_angle < -6:
                    # print("833L 右大旋转  turn001R < -6 ",Head_L_R_angle)
                    utils.act("turn001R")
                else:
                    print("836L bottom_angle < -3 需要小右转 turn001R ", top_angle)
                    utils.act("turn001R")
            elif -2.5 <= top_angle <= 2.5:  # 角度正确
                print("839L 角度合适")

                if topcenter_x > 255 or topcenter_x < 230:
                    if topcenter_x > 255:
                        print("843L 微微右移,", topcenter_x)
                        utils.act("Right3move")
                    elif topcenter_x < 230:
                        print("846L 微微左移,", topcenter_x)
                        utils.act("Left3move")

                else:
                    print("850L 位置合适")
                    break

def hole_edge_main(color):
    global HeadOrg_img, ChestOrg_img
    angle_ok_flag = False
    angle = 90
    dis = 0
    bottom_centreX = 0
    bottom_centreY = 0
    see = False
    dis_ok_count = 0
    headTURN = 0
    hole_flag = 0

    step = 1
    print("根据洞边缘调整位置....")
    while True:
        OrgFrame = HeadOrg_img.copy()
        x_start = 260
        blobs = OrgFrame[int(0):int(480), int(x_start):int(380)]  # 只对中间部分识别处理  Y , X
        handling = blobs.copy()
        frame_mask = blobs.copy()

        # 获取图像中心点坐标x, y
        center = []
        # 开始处理图像

        hsv = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        Imask = cv2.inRange(hsv, hole_color_range[color][0], hole_color_range[color][1])
        # Imask = cv2.erode(Imask, np.ones((3, 3), np.uint8), iterations=1)
        Imask = cv2.dilate(Imask, np.ones((3, 3), np.uint8), iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        Imask = cv2.morphologyEx(Imask, cv2.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv2.findContours(Imask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
        # cv2.imshow("opened",Imask)
        # print("len:",len(cnts))

        if len(contours) > 0:
            max_area = max(contours, key=cv2.contourArea)
            epsilon = 0.05 * cv2.arcLength(max_area, True)
            approx = cv2.approxPolyDP(max_area, epsilon, True)
            approx_list = list(approx)
            approx_after = []
            for i in range(len(approx_list)):
                approx_after.append(approx_list[i][0])
            approx_sort = sorted(approx_after, key=lambda x: x[1], reverse=True)
            # if approx_sort[0][0] > approx_sort[1][0]:
            #     approx_sort[0], approx_sort[1] = approx_sort[1], approx_sort[0]
            if len(approx_sort) == 4:
                bottom_line = (approx_sort[3], approx_sort[2])
                center_x = (bottom_line[1][0] + bottom_line[0][0]) / 2
                center_y = (bottom_line[1][1] + bottom_line[0][1]) / 2
            else:
                bottom_line = None

        else:
            bottom_line = None

        # 初始化
        L_R_angle = 0
        blackLine_L = [0, 0]
        blackLine_R = [0, 0]

        if bottom_line is not None:
            see = True
            if bottom_line[0][1] - bottom_line[1][1] == 0:
                angle = 90
            else:
                angle = - math.atan(
                    (bottom_line[1][1] - bottom_line[0][1]) / (bottom_line[1][0] - bottom_line[0][0])) * 180.0 / math.pi
            Ycenter = int((bottom_line[1][1] + bottom_line[0][1]) / 2)
            Xcenter = int((bottom_line[1][0] + bottom_line[0][0]) / 2)
            if bottom_line[1][1] > bottom_line[0][1]:
                blackLine_L = [bottom_line[1][0], bottom_line[1][1]]
                blackLine_R = [bottom_line[0][0], bottom_line[0][1]]
            else:
                blackLine_L = [bottom_line[0][0], bottom_line[0][1]]
                blackLine_R = [bottom_line[1][0], bottom_line[1][1]]
            cv2.circle(OrgFrame, (Xcenter + x_start, Ycenter), 10, (255, 255, 0), -1)  # 画出中心点

            if blackLine_L[0] == blackLine_R[0]:
                L_R_angle = 0
            else:
                L_R_angle = (-math.atan(
                    (blackLine_L[1] - blackLine_R[1]) / (blackLine_L[0] - blackLine_R[0])) * 180.0 / math.pi) - 4

            if Debug:
                cv2.circle(OrgFrame, (blackLine_L[0] + x_start, blackLine_L[1]), 5, [0, 255, 255], 2)
                cv2.circle(OrgFrame, (blackLine_R[0] + x_start, blackLine_R[1]), 5, [255, 0, 255], 2)
                cv2.line(OrgFrame, (blackLine_R[0] + x_start, blackLine_R[1]),
                         (blackLine_L[0] + x_start, blackLine_L[1]), (0, 255, 255), thickness=2)
                cv2.putText(OrgFrame, "L_R_angle:" + str(L_R_angle), (10, OrgFrame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Xcenter:" + str(Xcenter + x_start), (10, OrgFrame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(OrgFrame, "Ycenter:" + str(Ycenter), (200, OrgFrame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # cv2.drawContours(frame_mask, cnt_sum, -1, (255, 0, 255), 3)
                cv2.imwrite('./log/hole/'+utils.getlogtime()+'frame_mask.jpg', frame_mask)
                # cv2.imshow('black', Imask)
                cv2.imwrite('./log/hole/'+utils.getlogtime()+'OrgFrame.jpg', OrgFrame)
                cv2.waitKey(10)
        else:
            see = False

        # print(Ycenter)

        # 决策执行动作
        if step == 1:
            print("3266L 向右看 HeadTurnRight")
            utils.act("HeadTurnRight")
            time.sleep(1)  # timefftest
            step = 2
        elif step == 2:
            if not see:  # not see the edge
                # cv2.destroyAllWindows()
                print("3273L 右侧看不到边缘 左侧移 Left3move")
                utils.act("Left3move")
            else:  # 0
                if L_R_angle > 1.5:
                    if L_R_angle > 7:
                        headTURN += 1
                        print("3279L 左大旋转 turn001L ", L_R_angle)
                        utils.act("turn001L")

                    else:
                        print("3283L 左旋转 turn000L ", L_R_angle)
                        headTURN += 1
                        utils.act("turn000L")

                    # time.sleep(1)   # timefftest
                elif L_R_angle < -1.5:
                    if L_R_angle < -7:
                        headTURN += 1
                        print("3292L 右大旋转  turn001R ", L_R_angle)
                        utils.act("turn001R")

                    else:
                        print("3296L 右旋转  turn000R ", L_R_angle)
                        utils.act("turn000R")

                    # time.sleep(1)   # timefftest
                elif Ycenter >= 365:
                    if Ycenter > 390:
                        print("3303L 左大侧移 Left3move >440 ", Ycenter)
                        utils.act("Left3move")
                    else:
                        print("3306L 左侧移 Left02move > 365 ", Ycenter)
                        utils.act("Left02move")
                elif Ycenter < 355:
                    print("3309L 右侧移 Right02move <400 ", Ycenter)
                    utils.act("Right02move")
                else:
                    print("3312L 右看 X位置ok")
                    utils.act("fastForward03")
                    # utils.act("Left02move")
                    print("向前一步")
                    utils.act("forwardSlow0403")
                    # utils.act("forwardSlow0403")
                    # utils.act("forwardSlow0403")
                    utils.act("Stand")
                    step = 3
                    # cv2.destroyAllWindows()


        elif step == 3:
            if not see:  # not see the edge
                # cv2.destroyAllWindows()
                print("3327L 右侧看不到边缘 左侧移 Left3move")
                step == 5
            else:  # 0
                if L_R_angle > 2:
                    if L_R_angle > 7:
                        print("3332L 左旋转 turn001L ", L_R_angle)
                        utils.act("turn001L")
                    else:
                        print("3335L 左旋转 turn000L ", L_R_angle)
                        utils.act("turn000L")

                    # time.sleep(1)   # timefftest
                elif L_R_angle < -2:
                    if L_R_angle < -7:
                        print("3341L 右旋转  turn001R ", L_R_angle)
                        utils.act("turn001R")
                    else:
                        print("3344L 右旋转  turn000R ", L_R_angle)
                        utils.act("turn000R")
                    # time.sleep(1)   # timefftest
                else:
                    print("666L 右看 X位置ok")
                    step = 4

        elif step == 4:
            print("3352L 右侧看到绿色边缘 右侧移 Right3move")
            utils.act("Right3move")
            utils.act("Right3move")
            utils.act("Right3move")
            # utils.act("Right3move")
            utils.act("HeadTurnMM")
            step = 5

        elif step == 5:
            print("过坑阶段结束")
            break

def hole_edge(color):
    edge_angle_chest(color)  # 调整好角度与距离
    while (1):
        src = ChestOrg_img.copy()
        src = src[int(100):int(400), int(50):int(500)]
        src = cv2.GaussianBlur(src, (5, 5), 0)
        hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, hole_color_range[color][0], hole_color_range[color][1])

        mask2 = cv2.erode(mask, None, iterations=5)
        mask1 = cv2.dilate(mask2, None, iterations=8)

        contours2, hierarchy2 = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # dts = cv2.drawContours(ChestOrg_img.copy(), contours2, 0,(0, 0, 255),cv2.FILLED)
        # cv2.imwrite('./test.jpg', dts)
        if len(contours2) >= 2:
            print("3146L 仍然看得到内轮廓，向前走 forwardSlow0403")
            utils.act("forwardSlow0403")

        else:
            print("已经迈进，正式进入过坑阶段")
            utils.act("Stand")
            if color == 'blue_hole_chest':
                hole_edge_main('blue_hole_chest')
                break
            elif color == 'green_hole_chest':
                hole_edge_main('green_hole_chest')
                break

def main(color):
    time.sleep(3)
    while True:
        recognize = hole_recognize(color)
        print("recognize = ", recognize)
        if recognize:
            hole_edge(color)
            break
        time.sleep(0.5)
    return True


if __name__ == '__main__':
    rospy.init_node('runningrobot')
    edge_angle_chest("gree_hole_chest")