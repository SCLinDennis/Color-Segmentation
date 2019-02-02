#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:58:10 2019

@author: DennisLin
"""

import os
import cv2
import pickle 
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

ROOT = '/Users/DennisLin/Documents/Python/ECE276A/ECE276A_HW1/'
MASK = ROOT + 'mask/'
IMAGE = ROOT + 'trainset/'
os.chdir(IMAGE)

file = open('../train_data.pkl', 'rb')
train_data = pickle.load(file)
file.close()

parameters_dict = {}
directory = os.fsencode(IMAGE) 
for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename[-4:] != '.png':
        continue
    print('Now processing ' + filename)
    test_prefix = filename[:-4]
    
    blue = []
    blue_like = []
    non = []
    for key, value in train_data.items():
        feature, label = value
        (H, W) = feature.shape
        for h in range(H):
            if label[h][0] == 1:
                blue.append(feature[h, :])
            elif label[h][1] == 1:
                blue_like.append(feature[h, :])
            else:
                non.append(feature[h, :])
    prior_blue = len(blue) / (len(blue) + len(non))
    prior_non = len(non) / (len(blue) + len(non))
    
    prior_blue2 = len(blue) / (len(blue) + len(blue_like))
    prior_bluelike = len(blue_like) / (len(blue) + len(blue_like))
    
    blue = np.array(blue)
    blue_like = np.array(blue_like)
    non = np.array(non)
    
    mean_blue = np.mean(blue, 0)
    var_blue = np.diag(np.var(blue, 0))
    mean_non = np.mean(non, 0)
    var_non = np.diag(np.var(non, 0))
    mean_bluelike = np.mean(blue_like, 0)
    var_bluelike = np.diag(np.var(blue_like, 0))
    parameters_dict[test_prefix] = [mean_blue, var_blue, mean_non, var_non, mean_bluelike, var_bluelike]

file = open('../parameters.pkl', 'wb')
pickle.dump(parameters_dict, file)
file.close()
