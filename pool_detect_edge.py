# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import cv2


def main():
#    img = cv2.imread("outdoor-pool_lines.jpg")
    img = cv2.imread('./Pool_Images/pool4.jpg')
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(img[:,:,0],(5,5),0)
    edges = cv2.Canny(blur,100,200,3)
    
#    plt.subplot(1,4,1)
#    plt.imshow(img[:,:,0])
    plt.imshow(edges)
#    plt.imshow(blur)
    plt.show()

if __name__ == '__main__':
    main()
