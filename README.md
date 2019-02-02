# Shih-Chen Lin <span style="color:red">(A53276247)</span>

# Project 1 / Color Segmentation
In this project, we implement a probabilistic color model from image data and use it to segment unseen images, detect a blue barrel, and draw a bounding box around it. Given a set of training images, we manually label examples of different colors. Moreover, we build two single Gaussian classifiers, one for blue and non-blue and the other for dark blue and light blue. By setting some threshold, we could obtain the bounding box of a detected blue barrel given mask of the image.

## Requirement
- Python 3.6
- OpenCV >= 3.4 
- Numpy >= 1.14
- Matplotlib >= 2.2

## Usage 
1. To inference the color segementation model and blue barrel detector
    ```python
    my_detector = BarrelDetector()
    img = cv2.imread(IMAGE + filename)
    mask_img = my_detector.segment_image(img)
    boxes = my_detector.get_bounding_box(img)
    ```
Otherwise, you might want to train the color segmentation and barrel detector by yourselves. Then please follow the steps below.
1. Run `pip3 install -r requirements.txt`
2. Run `pip3 install git+https://github.com/jdoepfert/roipoly.py`.
3. Please insert the training data path to `DIR` in `label.py`. Then run it to annotate the training images and save the mask as `.npy` files in `./mask/`.
    ```python
    python label.py
    ```
4. Extract RGB features from training data, and save it as pickle file `train_data.pkl`.
    ```python
    python extract_rgb.py
    ```
5. Calculate the parameters of single Gaussian classification model using maximum likelihood estimation, and save them as `parameters.pkl`.
    ```python 
    python preprocess.py
    ```
6. Run leave one out cross validation to see the mIoU.
    ```python
    python validation.py
    ```
