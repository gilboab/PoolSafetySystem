
import cv2
import numpy as np

# put your image here
frame = cv2.imread('./Pool_Images/pool20.jpg')
#frame = cv2.imread('./Doron_Pool/d1.jpg')
#frame = cv2.imread('./My_Pool/img2.jpg')
# This drives the program into an infinite loop.

#while (1):
for k in range (0,1):
    # Converts images from BGR to HSV
    cv2.imshow('frame', frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    # define range of blue color in HSV - 90 - 110 is the pool color you can tweak it
    lower_blue = np.array([60, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blur = cv2.GaussianBlur(hsv[:,:,0],(5,5),0)
    edges = cv2.Canny(blur,100,200,3)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow('edges',edges)
    #cv2.imshow('mask', mask)
    img = mask
    # threshold image
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    # find contours
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour
    i=0
    largest_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > largest_area):
            largest_area = area
            largest_contour_index = i
        i=i+1
    hull = cv2.convexHull(contours[largest_contour_index])
    new_frame = frame.copy()
    for i in range (0,np.shape(frame)[0]):
        for j in range (0,np.shape(frame)[1]):
            if cv2.pointPolygonTest(hull, (j,i), False )<0:
                new_frame[i,j,:] = (0,0,0)
    #cv2.drawContours(new_frame, [hull], 0, (0, 0, 255), 3)
    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour
    i=0
    largest_area=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > largest_area):
            largest_area = area
            largest_contour_index = i
        i=i+1
    #cv2.imshow('mask',mask)

    pct = 0.005
    epsilon = pct * cv2.arcLength(contours[largest_contour_index], True)
    approx = cv2.approxPolyDP(contours[largest_contour_index], epsilon, True)
    x = len(approx)
    while x>200:
        epsilon = pct * cv2.arcLength(contours[largest_contour_index], True)
        approx = cv2.approxPolyDP(contours[largest_contour_index], epsilon, True)
        x=len(approx)
        pct=pct+0.005

    out = frame.copy()
    hull_img = frame.copy()
    hull = cv2.convexHull(approx)
    
    cv2.drawContours(out, [approx], 0, (0, 0, 255), 3)
    cv2.drawContours(hull_img, [hull], 0, (0, 0, 255), 3)
    # display output
    cv2.imshow('hull', hull_img)
    cv2.imwrite('./Pool_Images/pool22_modified.jpg',hull_img)
    cv2.imshow('image', out)

cv2.waitKey(0)
cv2.destroyAllWindows()






