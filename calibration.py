
import cv2
import numpy as np
from matplotlib import pyplot as plt

#img1 = cv2.imread('./Doron_Pool/d1.jpg')
#img1 = cv2.imread('./My_Pool/c2.jpg')
img1 = cv2.imread('./My_pool3/c2.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Living room
#u1 = np.array([475, 465, 454, 443, 442, 455, 466, 477, 491, 508, 526, 525, 492, 509, 480, 486, 493, 474, 504, 457])
#v1 = np.array([349, 335, 318, 302, 341, 299, 316, 294, 334, 318, 301, 340, 297, 299, 355, 363, 370, 367, 366, 363])
# pool2 c1
#u1 = np.array([513, 497, 480, 461, 460, 480, 498, 517, 533, 553, 574, 572, 534, 554, 514, 514, 514, 494, 533, 475])
#v1 = np.array([489, 472, 454, 436, 484, 430, 448, 421, 475, 460, 444, 493, 427, 434, 504, 522, 542, 530, 533, 519])
#  pool2 c2
#u1 = np.array([320, 296, 271, 245, 247, 270, 296, 319, 328, 335, 432, 342, 326, 334, 303, 284, 263, 257, 290, 252])
#v1 = np.array([498, 479, 459, 438, 487, 435, 455, 428, 485, 469, 453, 505, 435, 443, 512, 529, 547, 534, 541, 522])
# pool3 c1
#u1 = np.array([195, 176, 155, 133, 134, 155, 176, 195, 204, 213, 222, 222, 203, 212, 184, 171, 157, 149, 180, 142])
#v1 = np.array([400, 385, 370, 354, 395, 350, 366, 342, 388, 375, 361, 404, 348, 355, 413, 428, 445, 434, 437, 424])
# pool3 c2
u1 = np.array([429, 421, 411, 401, 398, 412, 422, 433, 448, 470, 492, 488, 452, 471, 437, 446, 458, 436, 468, 416])
v1 = np.array([411, 400, 387, 373, 414, 366, 379, 354, 399, 386, 372, 413, 360, 366, 425, 442, 461, 450, 451, 442])


P = np.array([(0,0,0,1), (2,2,0,1), (4,4,0,1), (6,6,0,1),(6,2,0,1),(4,6,0,1),(2,4,0,1),(0,6,0,1),(0,2,2,1),(0,4,4,1),(0,6,6,1),(0,2,6,1),(0,6,2,1),(0,6,4,1),(2,0,2,1),(4,0,4,1),(6,0,6,1),(6,0,4,1),(4,0,6,1),(6,0,2,1)])
P1_mat = np.zeros((40,12)) # for 11 calibration points
P2_mat = np.zeros((40,12)) # for 11 calibration points
for i in range (0,20):
    P1_mat[2*i,0]    =  P[i,0]
    P1_mat[2*i,1]    =  P[i,1]
    P1_mat[2*i,2]    =  P[i,2]
    P1_mat[2*i,3]    =  P[i,3]
    P1_mat[2*i,8]    =  -u1[i]*P[i,0]
    P1_mat[2*i,9]    =  -u1[i]*P[i,1]
    P1_mat[2*i,10]   =  -u1[i]*P[i,2]
    P1_mat[2*i,11]   =  -u1[i]*P[i,3]
    P1_mat[2*i+1,4]  =  P[i,0]
    P1_mat[2*i+1,5]  =  P[i,1]
    P1_mat[2*i+1,6]  =  P[i,2]
    P1_mat[2*i+1,7]  =  P[i,3]
    P1_mat[2*i+1,8]  =  -v1[i]*P[i,0]
    P1_mat[2*i+1,9]  =  -v1[i]*P[i,1]
    P1_mat[2*i+1,10] =  -v1[i]*P[i,2]
    P1_mat[2*i+1,11] =  -v1[i]*P[i,3]
U,S,Vt = np.linalg.svd(P1_mat)
V=Vt.T
print (np.shape(V))
m = V[:,11]
M1 = np.reshape(m,(3,4))
print (M1)
P_test = np.array([(-100,50,-100,1),(-100,20,-100,1),(-70,50,-100,1),(-70,20,-100,1),(-100,50,-70,1),(-100,20,-70,1),(-70,50,-70,1),(-70,20,-70,1)])
p1_test = M1.dot(P_test.T)
x1_test = np.zeros(8)
y1_test = np.zeros(8)
for i in range (0,8):
    x1_test[i] = np.round(p1_test[0,i]/p1_test[2,i])
    y1_test[i] = np.round(p1_test[1,i]/p1_test[2,i])
x1_test = x1_test.astype(int)
y1_test = y1_test.astype(int)
for i in range (0,8):
    for j in range (0,8):
        if np.sqrt((P_test[i,0]-P_test[j,0])**2 + (P_test[i,1]-P_test[j,1])**2 + (P_test[i,2]-P_test[j,2])**2) < 31:
            cv2.line(img1,(x1_test[i],y1_test[i]),(x1_test[j],y1_test[j]), (255,0,0),2, -1)
    
#    cv2.circle(img1,(x1_test,y1_test), 4, (255,0,0), -1)
#p1_test = M1.dot(P.T)
#print(p1_test)
#x1_test = int(np.round(p1_test[0]/p1_test[2]))
#y1_test = int(np.round(p1_test[1]/p1_test[2]))
#print(np.shape(p1_test))
#cv2.circle(img1,(x1_test,y1_test), 4, (255,0,0), -1)
#for i in range (0,20):
#    x1_test = int(np.round(p1_test[0,i]/p1_test[2,i]))
#    y1_test = int(np.round(p1_test[1,i]/p1_test[2,i]))
#    cv2.circle(img1,(x1_test,y1_test), 4, (255,0,0), -1)
plt.imshow(img1)
plt.show()






