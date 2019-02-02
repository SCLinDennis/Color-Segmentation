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
#%%
class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.MIN_AREA = 300
        self.prior_blue2 = 0.3
        self.prior_bluelike = 0.7
        self.brightness = 80
        
        
        file = open('./parameters.pkl', 'rb')
        self.parameters_dict = pickle.load(file)
        file.close()

    def segment_image(self, img, filename):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        # YOUR CODE HERE
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255
        test_prefix = filename[:-4]
        mask_img = np.zeros(np.shape(img)[:2])
        (H, W, C) = img.shape
        mean_blue, var_blue, mean_non, var_non, mean_bluelike, var_bluelike = self.parameters_dict[test_prefix]
        test_image = []
        count = []
        for i in range(0, H, 1):
            for j in range(0, Ｗ, 1):
                grid = img[i:i+1, j:j+1]
                test_image.append(grid)
                count.append((i, j))
        test_image = np.array(test_image)
        mvn_blue = multivariate_normal.pdf(test_image, mean_blue, var_blue)*0.4        
        mvn_non = multivariate_normal.pdf(test_image, mean_non, var_non)*0.6
        out = mvn_blue>mvn_non
        mvn_blue = multivariate_normal.pdf(test_image[out], mean_blue, var_blue)*self.prior_blue2
        mvn_bluelike = multivariate_normal.pdf(test_image[out], mean_bluelike, var_bluelike)*self.prior_bluelike
        out[out==True] = mvn_blue>mvn_bluelike
        count = np.array(count)
        for i, j in count[np.where(out == True)]:
            mask_img[i:i+1, j:j+1] = 1
        mask_img = cv2.dilate(mask_img, None, iterations=1)
        return mask_img

    def get_bounding_box(self, img, filename):
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
        img_pre = self.segment_image(img, filename)
        img_pre = np.uint8(img_pre)
        
        contours, hierachy = cv2.findContours(img_pre.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        if len(contours) > 0:
            for contour in contours:
                if self.barrel_checker(contour):
                    x, y, w, h = cv2.boundingRect(contour)    
                    rect.append([x, y, w, h])
            rect.sort(key=lambda x: x[1])
        return rect

    def getBrightness(self, img):
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(img_HSV[:, :, 2])

    def brigher(self, img):
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_HSV)
        V[np.where(V<100)] += 80
        img_HSV = cv2.merge((H, S, V))
        return cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    
    def barrel_checker(self, contour):
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if area > self.MIN_AREA and h/w > 1:
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
ROOT = '/Users/DennisLin/Documents/Python/ECE276A/ECE276A_HW1/'
MASK = ROOT + 'mask/'
IMAGE = ROOT + 'trainset/'
os.chdir(ROOT)

if __name__ == '__main__':    
    folder = "trainset"
    my_detector = BarrelDetector()
    IoU = 0
    valid = 0
    for filename in sorted(os.listdir(folder)):
        prefix = filename[:-4]
        if filename[-4:] != '.png':
            continue

        # read one test image

        img = cv2.imread(IMAGE + filename)
        label = np.load(MASK + prefix + 'mask1.npy')
        segment = my_detector.segment_image(img, filename)
        rect = my_detector.get_bounding_box(img, filename)
        score = get_IoU(label, rect)
        if score > 0.5:
            valid += 1
        IoU += score
        print('IoU of ' + prefix + '.png is ' + str(score))
        for (x, y, w, h) in rect:
            print((x, y, w, h))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('image', segment)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(x,y, w, h)
    print('Total passed case:'+ str(valid)) 
    print('mIoU is ' + str(IoU/45))
        

