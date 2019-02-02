#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 23:31:21 2019

@author: DennisLin

input: mask.npy
output: features
"""

import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

ROOT = '/Users/DennisLin/Documents/Python/ECE276A/ECE276A_HW1/'
MASK1 = ROOT + '/mask/label/'
MASK2 = ROOT + '/mask/otherblue/'
IMAGE = ROOT + 'trainset/'
NUM_PADDING = 4
NUM_STRIDE = 4

def brigher(img):
    img = img*255
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_HSV)
    V[np.where(V<150)] += 50
    img_HSV = cv2.merge((H, S, V))
    return cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)

os.chdir(IMAGE)
train_data = {}
directory = os.fsencode(IMAGE) 
for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    print('Now processing ' + filename)
    if filename[-4:] != '.png':
        continue
    #preprocess the mask
    prefix = filename[:-4]
    mask2 = np.load(MASK2 + 'otherblue' + str(prefix)+ '.npy')

    mask1 = np.load(MASK1 + 'label' + str(prefix) + '.npy')
    
    #preprocess the image
    img = plt.imread(filename)
    img = brigher(img)/255
    (H, W, C) = img.shape
    img_pad = np.zeros((H+2*NUM_PADDING, W+2*NUM_PADDING, C))
    
    for c in range(C):
        img_pad[:, :, c] = np.pad(img[:, :, c], (NUM_PADDING, NUM_PADDING), 'edge')

    features = np.zeros((H*W//(NUM_STRIDE**2), C))
    labels = np.zeros((features.shape[0], 2))
    count = 0
    for i in range(0, img.shape[0]-8, NUM_STRIDE):
        for j in range(0, img.shape[1]-8, NUM_STRIDE):
            grid = img_pad[i:i+8, j:j+8]
            label1 = mask1[i, j]
            label2 = mask2[i, j]
            for c in range(C):
                features[count, c] = np.mean(grid[:, :, c])
            labels[count, :] = np.array([label1, label2])
            count += 1
    train_data[prefix] = [features, labels]

file = open('../train_data.pkl', 'wb')
pickle.dump(train_data, file)
file.close()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#%%
