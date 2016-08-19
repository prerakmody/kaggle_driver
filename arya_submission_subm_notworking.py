
# coding: utf-8

# In[1]:
"""
Get data from here: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
"""

import cv2, cPickle, os, glob, math, datetime, time
import numpy as np
from collections import Counter

""" RUN THIS JUST ONCE TO CREATE A CACHE OF DATA (Train/Test)"""
class getImgData(object):
    def __init__(self, path_file, data_type, path_folder='mycnn_cache', no_classes=10):
        self.data_type = data_type
        self.no_classes = no_classes
        self.cache_path = os.path.join(path_folder, path_file)
        print 'Cache Path:',self.cache_path

    def get_data(self):
        if self.data_type == "train":
            if not os.path.isfile(self.cache_path):
                train_data, train_target = self.load_train(self.no_classes)
                print 'Caching Train ...'; t1 = time.time()
                self.cache_data((train_data, train_target), self.cache_path)
                print 'Cached ... ',time.time() - t1,'s' 
            else:
                print('Restore train from cache!'); t1 = time.time()
                (train_data, train_target) = self.restore_data(self.cache_path)
                print 'Restored ... ', time.time() - t1,'s'   
            return train_data, train_target
        elif self.data_type == 'test':
            if not os.path.isfile(self.cache_path):
                test_data, test_id = self.load_test()
                self.cache_data((test_data, test_id), self.cache_path)
            else:
                print('Restore test from cache!'); t1 = time.time()
                (test_data, test_id) = self.restore_data(self.cache_path)
                print 'Restored test data ... ', time.time() - t1,'s'        
            return test_data, test_id

    def get_im(self, path):
        img = cv2.imread(path, 0)
        resized = cv2.resize(img, (128, 96))
        return resized

    def load_train(self, no_classes):
        X_train = []
        y_train = []
        print('No cache data. Read train images')
        for j in range(no_classes):  #this would have been 10
            print('Load folder c{}/c{}'.format(j, no_classes))
            path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
            files = glob.glob(path)
            for i,fl in enumerate(files):
                img = self.get_im(fl)
                X_train.append(img)
                y_train.append(j)
        return X_train, y_train

    def load_test(self):
        path = os.path.join('imgs', 'test', '*.jpg')
        files = glob.glob(path)
        print('No cache data.Read test images: Len: {}'.format(len(files)))
        X_test = []
        X_test_id = [] #contains the file path
        total = 0
        for i,fl in enumerate(files):
            if i>100 and i%100 == 0: break
            flbase = os.path.basename(fl)
            img = self.get_im(fl)
            X_test.append(img)
            X_test_id.append(flbase)
            total += 1
            if total%1000 == 0:
                print('Image:{}/{}'.format(total,len(files)))

        return X_test, X_test_id

    def cache_data(self, data, path):
        if os.path.isdir(os.path.dirname(path)):
            file = open(path, 'wb')
            cPickle.dump(data, file,2)  #use cPickle and protocol_version:2
            file.close()
        else:
            print('Directory doesnt exists:',path)

    def restore_data(self, path):
        data = dict()
        if os.path.isfile(path):
            file = open(path, 'rb')
            data = cPickle.load(file)
        else:
            print 'File not found'
        return data

train_data, train_target = getImgData('train.dat', 'train',no_classes=1).get_data()
# test_data, test_target = getImgData('test.dat', 'test').get_data()   #shall call this in the end


# In[2]:
print '\n Importing theano ... \n'
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from sklearn.cross_validation import train_test_split


# In[3]:

""" This does both convolution and pooling """
class ConvPoolLayer(object):
    def __init__(self, rng, input, image_shape, filter_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /  np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX
                ), borrow=True)
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv.conv2d(input=input, filters=self.W,
                    filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        self.output = theano.tensor.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

    
    def return_output(self):
        return self.output

""" This layer comes between the Convolution and Output layer"""
class HiddenLayer(object):
    def __init__(self, rng, srng, input, n_in, n_out, is_train, W=None, b=None, activation=theano.tensor.nnet.relu, p=0.5):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform( low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)),
                                  dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
        # DropOut: multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = self.drop(np.cast[theano.config.floatX](1./p) * self.output, p, srng = srng)
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, self.output)
        
        # parameters of the model
        self.params = [self.W, self.b]

    def drop(self, input, p, srng): 
        mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask

""" Final Output layer using softmax """
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)   #(n,n_classes)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)                   #(n,1)  
        self.params = [self.W, self.b]

    def errors(self, y):
        # simple dimension check
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # return (-y * T.log(self.y_pred) - (1-y)*T.log(1 - self.y_pred)).mean()
        # print 'Y:',y.ndim, ' y_pred:', self.y_pred.ndim, ' p_y_given_x:', self.p_y_given_x.ndim      #1 ,1, 2
        # print 'Y:',type(y), ' y_pred:', type(self.y_pred), ' p_y_given_x:', type(self.p_y_given_x)   #<class 'theano.tensor.var.TensorVariable'>

    def predict(self):
        return self.y_pred


# In[5]:


""" This class is called in main and stitches all the above classes together. Make Changes here.  """
class CNN(object):

    def __init__(self, train_data, rng, srng, test_data=[], batch_size = 128, nb_classes = 10, nb_epoch = 10, 
                 img_rows = 96,  img_cols = 128, 
                nb_filters = [32,64], nb_pool = 2, nb_conv = 3, init_learning_rate=0.025, validate_size = 0.3):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nb_filters = nb_filters
        self.nb_pool = nb_pool
        self.nb_conv = nb_conv    #kernel size
        self.rng = rng
        self.srng = srng
        self.init_learning_rate = init_learning_rate

        self.train_data = np.array(train_data, dtype=np.uint8)
        self.train_target = np.array(train_target, dtype=np.uint8)
        self.train_data = self.train_data.astype('float32')
        self.train_data /= 255
        self.validate_size = validate_size
        
    def createCNN(self):
        index = T.iscalar('index')
        x = T.matrix('x')
        y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction

        layer0_input = x.reshape((self.batch_size, 1, self.img_rows, self.img_cols))  #1 - depth, 3 for RGB
        #(128, 1, 96, 128) -> (128, 32, 94, 126) -> (128 ,32, 47, 63)
        print 'Conv Layer0:',self.batch_size, 1, self.img_rows, self.img_cols
        layer0 = ConvPoolLayer(self.rng, layer0_input, (self.batch_size, 1, self.img_rows, self.img_cols), 
            filter_shape=(self.nb_filters[0], 1, self.nb_conv, self.nb_conv), poolsize=(self.nb_pool, self.nb_pool))

        #(128, 32, 47, 63) --> (128, 32, 45, 61) --> (128, 32, 22, 30)
        new_rows = int((self.img_rows - self.nb_conv + 1)/2) #since stride is 1
        new_cols = int((self.img_cols -self.nb_conv + 1)/2)
        print 'Conv Layer1:', self.batch_size, self.nb_filters[0], new_rows, new_cols
        layer1 = ConvPoolLayer(self.rng, input=layer0.output, image_shape=(self.batch_size, self.nb_filters[0], new_rows, new_cols), 
                filter_shape=(self.nb_filters[0], self.nb_filters[0], self.nb_conv, self.nb_conv), poolsize=(self.nb_pool, self.nb_pool))

        # (128, 32, 22, 30) -> (128, 32, 20, 28) -> (128, 64, 10, 14)
        new_rows = int((new_rows - self.nb_conv + 1)/2)
        new_cols = int((new_cols - self.nb_conv + 1)/2)
        print 'Conv Layer2:', self.batch_size, self.nb_filters[0], new_rows, new_cols
        layer2 = ConvPoolLayer(self.rng, input=layer1.output, image_shape = (self.batch_size, self.nb_filters[0], new_rows, new_cols),
                filter_shape=(self.nb_filters[1], self.nb_filters[0], self.nb_conv, self.nb_conv), poolsize=(self.nb_pool, self.nb_pool))

        # (64, 10, 14) -> (64,10,14)
        new_rows = int((new_rows - self.nb_conv + 1)/2)
        new_cols = int((new_cols - self.nb_conv + 1)/2)
        layer3_input = layer2.output.flatten(2)   #convert to two dimnesions
        print 'Hidden Layer3:',self.nb_filters[1], new_rows, new_cols, '(',self.nb_filters[1] * new_rows * new_cols,')'
        layer3 = HiddenLayer(self.rng, self.srng, input=layer3_input, n_in = self.nb_filters[1] * new_rows * new_cols, 
                             n_out = self.nb_filters[1] * new_rows * new_cols, 
            is_train = is_train, activation = theano.tensor.nnet.relu)        

        #(64, 10, 14) -> (10, 14)
        print 'Hidden layer4:',self.nb_filters[1],new_rows,new_cols,'(',self.nb_filters[1] * new_rows * new_cols,')'
        layer4 = HiddenLayer(self.rng, self.srng, input=layer3.output, n_in= self.nb_filters[1] * new_rows * new_cols, 
                             n_out = new_rows * new_cols, 
            is_train = is_train, activation = theano.tensor.nnet.relu)
    
        print 'Logistic layer5:',new_rows, new_cols, '(',new_rows * new_cols,')'
        layer5 = LogisticRegression(input=layer4.output, n_in = new_rows * new_cols, n_out = self.nb_classes)

        cost = layer5.negative_log_likelihood(y)

        self.params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        
        grads = T.grad(cost, self.params)
        
        updates = [];
        self.learning_rate = theano.shared(np.cast[theano.config.floatX](self.init_learning_rate) )
        for param_i, grad_i in zip(self.params, grads):
            updates.append((param_i, param_i - self.learning_rate * grad_i))
        
        ################################### THEANO FUNCTIONS ##############################
        print 'Splitting train data ...'
        X_train, X_val, Y_train, Y_val = train_test_split(self.train_data, self.train_target, test_size = self.validate_size)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2])
        self.X_train_length = len(X_train)
        self.X_val_length = len(X_val)

        self.shared_train_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
        self.shared_train_y = T.cast(theano.shared(np.asarray(Y_train, dtype=theano.config.floatX), borrow=True), 'int32')

        self.shared_val_x = theano.shared(np.asarray(X_val, dtype=theano.config.floatX), borrow=True)
        self.shared_val_y = T.cast(theano.shared(np.asarray(Y_val, dtype=theano.config.floatX), borrow=True), 'int32')

        print 'Create theano train function ...'        
        self.batch_size = np.cast['int32'](self.batch_size)
        self.train_model = theano.function(
            inputs=[T.cast(index, dtype='int32')], outputs=cost, updates=updates,
            givens={
                is_train: np.cast['int32'](1),
                y: self.shared_train_y[index * self.batch_size: (index + np.cast['int32'](1)) * self.batch_size],
                x: self.shared_train_x[index * self.batch_size: (index + np.cast['int32'](1)) * self.batch_size]
                }
            )
        print 'Create theano validation function ...'
        self.val_model = theano.function(
            inputs=[T.cast(index, dtype='int32')], outputs=cost, updates=updates,
            givens={
                is_train: np.cast['int32'](1),
                y: self.shared_val_y[index * self.batch_size: (index + np.cast['int32'](1)) * self.batch_size],
                x: self.shared_val_x[index * self.batch_size: (index + np.cast['int32'](1)) * self.batch_size]
                }
            )
        
        print 'Creating theano test data function ...'
        self.test_model = theano.function(
            inputs=[x, is_train], outputs=layer5.p_y_given_x)

    def train(self):
        epoch = 0;
        while (epoch < self.nb_epoch):
            epoch = epoch + 1; 
            print "learning rate: ", self.learning_rate.get_value()
            for i,minibatch_index in enumerate(xrange(self.X_train_length/self.batch_size)):
                # print 'Train Batch:',i, len(xrange(self.X_train_length/self.batch_size)) 
                if i < len(xrange(self.X_train_length/self.batch_size)) - 1:
                    iter = (epoch - 1) * self.X_train_length/self.batch_size + minibatch_index
                    cost_ij = self.train_model(minibatch_index)
                    if iter % 20 == 0:
                        print '-------------------------Epoch:',epoch,' training@iter = ', iter, ' Cost:',cost_ij
                    

            print '\n Validate model ...'
            validation_losses = []
            for val_batch_idx in xrange(self.X_val_length/self.batch_size):
                if val_batch_idx < len(xrange(self.X_val_length/self.batch_size)) -1:
                    temp_loss = self.val_model(val_batch_idx)
                    validation_losses.append(temp_loss)
            print 'Epoch:',epoch, ' Loss:',np.mean(validation_losses)
        print '=============================================================================================Done With Training:' 


rng = np.random.RandomState(23455)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
batch_size = 64   #this can be increassed if he GPU has enough memory
neural_obj = CNN(train_data, rng, srng, batch_size = batch_size, nb_epoch = 1)
neural_obj.createCNN()
neural_obj.train()

neural_obj.shared_train_x.set_value([[]])
neural_obj.shared_val_x.set_value([[]])

try:
    file = open(os.path.join('mycnn_cache', 'params.dat'), 'wb')
    cPickle.dump(neural_obj.params, file,2)  #use cPickle and protocol_version:2
    file.close()
    print neural_obj.params
except Exception,e:
    print 'Model Params Caching Error'


print '\n Creating predictions ... \n'
######################################################## TEST DATASET ##############################################################
test_data, test_target = getImgData('test.dat', 'test').get_data()
test_data = np.array(test_data, dtype='float32')
test_data /= 255
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1]*test_data.shape[2])
test_target = np.array(test_target)


if len(test_data) % batch_size != 0:
    extras = (int(len(test_data)/batch_size) + 1)*batch_size - len(test_data)
    for i in range(extras):  #add this much dummy data
        test_data = np.append(test_data, np.random.rand(1,96*128), axis=0)
        test_target = np.append(test_target, ['-1'], axis=0)

print 'Final Len:',len(test_data), ' Mod',len(test_data)%batch_size

predictions = []
for test_batch_idx in xrange(len(test_data)/batch_size):
    print 'TestIdx: ', test_batch_idx, '/',len(test_data)/batch_size
    tmp_data = test_data[test_batch_idx * batch_size: (test_batch_idx + 1) * batch_size].astype('float32')
    temp_pred = list(neural_obj.test_model(tmp_data, 1))
    predictions = predictions + temp_pred


#Create submission file
import pandas as pd
result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
result1.loc[:, 'img'] = pd.Series(test_target, index=result1.index)
result1.to_csv('submission_theano.csv', index=False)    




