import pylab, os, numpy
from matplotlib import pyplot as plt
from PIL import Image

path = os.path.join('imgs', 'train', 'c0', 'img_34.jpg')
# print path
path = r'imgs\\train\\c0\\img_34.jpg'
print path
img = Image.open(path)
img = numpy.asarray(img, dtype='float64') / 256.
print img.shape
print img.transpose(2, 0, 1).shape
img_ = img.transpose(2, 0, 1).reshape(1, img.shape[2], img.shape[0], img.shape[1])
print img_.shape

plt.imshow(img)
plt.savefig('expand_dataset/real.jpg')
# pylab.subplot(1, 3, 1); 
# pylab.axis('off'); pylab.imshow(img)
# pylab.gray();
# print 'Show....'
# pylab.show()