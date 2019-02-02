import os
import logging
import numpy as np
from matplotlib import pyplot as plt

from roipoly import RoiPoly



#DIR = YOUR TRAINING DATA DIRECTORY

os.chdir(DIR)
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)


directory = os.fsencode(DIR) 

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if not filename.endswith("png"): 
        continue
        # print(os.path.join(directory, filename))
    index = filename[:-4] 
    if index != '46':
        continue
    img = plt.imread(filename)
    # Show the image
    fig = plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title("left click: line segment         right click: close region")
    plt.show(block=False)
    

    # Let user draw first ROI
    roi1 = RoiPoly(color='b', fig=fig)
#%%    
    # Show the image with the first ROI
    fig = plt.figure()
    plt.imshow(img)
    plt.colorbar()
    savefig('./test.png')
    roi1.display_roi()
    plt.title('draw second ROI (other blue)')
    plt.show(block=False)
    
    # Let user draw second ROI
    roi2 = RoiPoly(color='r', fig=fig)

    #%%
    # Show the image with both ROIs and their mean values
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    [x.display_roi() for x in [roi1, roi2]]
    plt.title('The three ROIs')
    plt.show()
    
    # Show ROI masks
    img = img[:, :, 0]
    plt.imshow(roi1.get_mask(img) + roi2.get_mask(img))   
#    np.save('../mask/' + index + 'mask1.npy', roi1.get_mask(img))
#    np.save('../mask/' + index + 'mask2.npy', roi2.get_mask(img))
#    np.save('./mask3.npy', roi3.get_mask(img))
    plt.title('ROI masks of the three ROIs')
    plt.show()


