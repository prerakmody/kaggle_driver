from keras import backend as K
from keras.models import model_from_json
import os, cv2
import numpy as np
from matplotlib import pyplot as plt

def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

def get_image(path):
	img = cv2.resize(cv2.imread(path, 0), (128, 96))
	print img.shape
	img_ = img.reshape(1, 1, img.shape[0], img.shape[1])
	return img_

def save_img(data, layer_name, layer_id,path):
	plt.axis('off'); plt.gray()
	my_dpi = 96
	path = path.split('\\')[2] + '_' + path.split('\\')[3]
	for j in range(data[0].shape[0]):   #no. of filters
		plt.imshow(data[0, j, :, :])
		plt.savefig('Keras_Kaggle/_' + path + '_layer_' + str(layer_id) + layer_name + '_' + str(j+1) + '.jpg', dpi=my_dpi)

if __name__ == "__main__":
	model = read_model()
	# print model; print model.get_config(); print model.summary()

	path = 'imgs\\train\\c0\\img_34.jpg'
	img = get_image(r'imgs\\train\\c0\\img_34.jpg')
	w = model.get_weights()
	for i,each in enumerate(model.layers): 
		print '------------',i,') Layer: ',each, '   Shape:', img.shape
		if len(img.shape) == 4:
			layer_op = K.function([model.layers[i].input, K.learning_phase()], [model.layers[i+1].output])
			temp_op = layer_op([img,0])
			img = np.asarray(temp_op[0])  #test_mode = 0; train_mode = 1
			save_img(img, str(each).split('.')[3].split(' ')[0], i+1,path)
		else:
			print 'Done'
			break