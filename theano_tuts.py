import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy
import pylab, os, numpy
from matplotlib import pyplot as plt
from PIL import Image
from theano.tensor.signal import pool
rng = numpy.random.RandomState(123123)


def layer(no_filters, inp_layers, filter_x, filter_y, pool_x, pool_y, inp_img, layer_id):
	print '----------------------------------------------------------------------------------------'
	input = T.tensor4(name='input')
	w_shp = (no_filters, inp_layers, filter_x, filter_y)    #no_filters * inputmap_count * filter_x * filter_y
	w_bound = numpy.sqrt(inp_layers * filter_x * filter_y)  #total_no_weights for every output variable
	W = theano.shared(numpy.asarray( rng.uniform(low = -1.0/w_bound, high=1.0/w_bound, size = w_shp), dtype=input.dtype), name='W')
	#print 'Type of W:', type(W); print W.shape.eval(); print W[0].eval() #takes time to print

	b_shp = (no_filters,)
	b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high=.5, size = b_shp), dtype=input.dtype), name='b')

	conv_out = conv2d(input, W) #this is called as a symbolic expression
	output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
	f = theano.function([input], output)
	print 'Layer', layer_id,': Applying conv2d: ', inp_img.shape
	filtered_img = f(inp_img)
	save_images(layer_id, no_filters, filtered_img, 'conv2d')

	maxpool_shape = (pool_x, pool_y)
	pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
	fpool = theano.function([input], pool_out)
	print 'Layer', layer_id,': Pooling: ',filtered_img.shape 
	pooled_img = fpool(filtered_img)
	save_images(layer_id, no_filters, pooled_img, 'pool')

	return pooled_img

def get_image(path):
	img = Image.open(path)
	img = numpy.asarray(img, dtype='float32') / 256.
	img_ = img.transpose(2, 0, 1).reshape(1, img.shape[2], img.shape[0], img.shape[1])
	return img_

def save_images(layer_id, no_filters, img_4dtensor, name):
	plt.axis('off'); plt.gray()
	my_dpi = 96
	for i in range(no_filters): 
		#pass
		# plt.figure(figsize=(filtered_img[0][i].shape[0]/my_dpi, filtered_img[0][i].shape[1]/my_dpi), dpi=my_dpi)
		plt.imshow(img_4dtensor[0, i, :, :])
		plt.savefig('test_theano/layer' + str(layer_id) + '_' + str(i+1) + '_' +str(name) + '.jpg', dpi=my_dpi)

if __name__ == "__main__":
	no_filters = 12;
	inp_layers = 3; filter_x = 9; filter_y = 9;
	pool_x = 2; pool_y = 2;
	path = r'imgs\\train\\c0\\img_34.jpg'
	img_array = get_image(path)

	#Layer1
	layer_op = layer(no_filters, inp_layers, filter_x, filter_y, pool_x, pool_y, img_array, 1)
	print 'Layer 1 : Output: ', layer_op.shape
	
	#layer2
	layer_op = layer(24, layer_op[0].shape[0], 9, 9, 2, 2, layer_op, 2)
	print 'Layer 2 : Output: ', layer_op.shape

	#layer2
	layer_op = layer(48, layer_op[0].shape[0], 9, 9, 2, 2, layer_op, 2)
	print 'Layer 3 : Output: ', layer_op.shape