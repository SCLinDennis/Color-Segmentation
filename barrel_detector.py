#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 22:28:00 2019

@author: DennisLin
"""

'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
#from skimage.measure import label, regionprops

class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.NUM_PADDING = 4
        self.MIN_AREA = 300
        self.prior_blue2 = 0.7
        self.prior_bluelike = 0.3
        self.brightness = 80
        self.ratio = 1

        self.mean_blue = np.array([0.19664931, 0.37537237, 0.60549491])
        self.var_blue = np.diag(np.array([0.02508564, 0.01639323, 0.02467447]))
        self.mean_non = np.array([0.55110773, 0.51118504, 0.44241384])
        self.var_non = np.diag(np.array([0.03543071, 0.03377203, 0.04270499]))
        self.mean_bluelike = np.array([0.40692763, 0.50537949, 0.59870883])
        self.var_bluelike = np.diag(np.array([0.02172313, 0.01408661, 0.01606181]))

    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        img = self.brigher(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255
        mask_img = np.zeros(np.shape(img)[:2])
        (H, W, C) = img.shape
        img_pad = np.zeros((H+2*self.NUM_PADDING, W+2*self.NUM_PADDING, C))
        for c in range(C):
            img_pad[:, :, c] = np.pad(img[:, :, c], (self.NUM_PADDING, self.NUM_PADDING), 'edge')
        test_image = []
        count = []
        for i in range(0, H-8, 4):
            for j in range(0, Ｗ-8, 4):
                grid = img_pad[i:i+8, j:j+8]
                feature = []
                for c in range(C):
                    feature.append(np.mean(grid[:, :, c]))
                test_image.append(feature)
                count.append((i, j))
        test_image = np.array(test_image)
        mvn_blue = multivariate_normal.pdf(test_image, self.mean_blue, self.var_blue)*0.09                
        mvn_non = multivariate_normal.pdf(test_image, self.mean_non, self.var_non)*0.91
        out = mvn_blue>mvn_non
        mvn_blue = multivariate_normal.pdf(test_image[out], self.mean_blue, self.var_blue)*self.prior_blue2
        mvn_bluelike = multivariate_normal.pdf(test_image[out], self.mean_bluelike, self.var_bluelike)*self.prior_bluelike
        out[out==True] = mvn_blue>mvn_bluelike
        count = np.array(count)
        for i, j in count[np.where(out == True)]:
            mask_img[i:i+4, j:j+4] = 1
        mask_img = cv2.dilate(mask_img, None, iterations=2)
        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.
                
            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
        img_pre = self.segment_image(img)
        img_pre = cv2.erode(img_pre, None, iterations=2)
        img_pre = cv2.dilate(img_pre, None, iterations=2)
        img_pre = np.uint8(img_pre)
        
        contours, hierachy = cv2.findContours(img_pre.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        if len(contours) > 0:
            for contour in contours:
                if self.barrel_checker(contour):
                    x, y, w, h = cv2.boundingRect(contour)    
                    rect.append([x, y, x+w, y+h])
            rect.sort(key=lambda x: x[1])
        return rect

    def getBrightness(self, img):
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(img_HSV[:, :, 2])

    def brigher(self, img):
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_HSV)
        V[np.where(V<150)] += 50
        img_HSV = cv2.merge((H, S, V))
        return cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    
    def barrel_checker(self, contour):
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if area > self.MIN_AREA and h/w > self.ratio:
            return True
        return False

def get_IoU(label, rect):
    intersection = 0
    union = 0
    for (x, y, w, h) in rect:
        intersection += np.where(label[y:y+h+1, x:x+w+1] == True)[0].shape[0]
        union += w*h
    union = union + np.where(label == True)[0].shape[0] - intersection
    return intersection / union


        #Display results:
        #(1) Segmented images
        #    mask_img = my_detector.segment_image(img)
        #(2) Barrel bounding box
        #    boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope

