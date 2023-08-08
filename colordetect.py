import cv2
import numpy as np
# import myFunc


def nothing(x):
    pass


cv2.namedWindow('Track')
cv2.resizeWindow('Track', (640, 320))
cv2.createTrackbar('H Min', 'Track', 0, 255, nothing)
cv2.createTrackbar('H Max', 'Track', 255, 255, nothing)
cv2.createTrackbar('S Min', 'Track', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Track', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Track', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Track', 255, 255, nothing)

img = cv2.imread("3.jpg")
img = cv2.resize(img, (640, 480), cv2.INTER_LINEAR)

while 1:
    # img = cv2.resize(cv2.imread('test2.jpg'),(320,240))

    # _,img = cap.read()

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('H Min', 'Track')
    h_max = cv2.getTrackbarPos('H Max', 'Track')
    s_min = cv2.getTrackbarPos('S Min', 'Track')
    s_max = cv2.getTrackbarPos('S Max', 'Track')
    v_min = cv2.getTrackbarPos('V Min', 'Track')
    v_max = cv2.getTrackbarPos('V Max', 'Track')

    low = np.array([h_min, s_min, v_min])
    up = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, low, up)

    res = cv2.bitwise_and(img, img, mask=mask)

    # show = myFunc.stackImages(0.6,([img,imgHSV],[mask,res]))
    cv2.imshow('figure', res)
    print(f"({h_min},{s_min},{v_min}),({h_max},{s_max},{v_max})")
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
