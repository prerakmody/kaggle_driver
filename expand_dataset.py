import cv2
import time
from matplotlib import pyplot as plt
import numpy as np

def get_im(path,w,h):
    # img = cv2.imread(path,0)
    img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    resized = cv2.resize(img, (w, h))
    return resized

w,h = 128,96
dirr = 'C:\Users\prerakmody\Desktop\RK\Kaggle\imgs\\train\c0\\'
imgname = 'img_34.jpg'
img = get_im(dirr+imgname, w,h)

print 'Original Shape:',img.shape
plt.imshow(img)
plt.savefig('expand_dataset/' + imgname)


#Apply 2 transformations (horizontal - Up / Down) 
#Horizonatal UP
new_img = np.roll(img, -1, 0)  #img, d, axis   axis=0, horizontal axis
new_img[h-1,:] = np.zeros(w)
plt.imshow(new_img)
plt.savefig('expand_dataset/' + imgname.split('.')[0] + '_hor_up_padded.png')
#Horizonatal DOWN
new_img = np.roll(img, 1, 0)  #img, d, axis   axis=0, horizontal axis
new_img[0,:] = np.zeros(w)
plt.imshow(new_img)
plt.savefig('expand_dataset/' + imgname.split('.')[0] + '_hor_down_padded.png')

#Apply 2 transformations (Vertical - Left / Right) 
#Vertical Left
new_img = np.roll(img, -1, 1)  #img, d, axis   axis=1, vertical axis
new_img[:,w-1] = np.zeros(h)
plt.imshow(new_img)
plt.savefig('expand_dataset/' + imgname.split('.')[0] + '_ver_left_padded.png')
#Verticl Right
new_img = np.roll(img, 1, 1)  #img, d, axis   axis=1, vertical axis
new_img[:,0] = np.zeros(h)
plt.imshow(new_img)
plt.savefig('expand_dataset/' + imgname.split('.')[0] +'_ver_right_padded.png')

# new_img[0, :] = np.zeros(128)	
# print 'Rolled:',new_img.shape
# print '=============================='
# print new_img
# ###########################################
#