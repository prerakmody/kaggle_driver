
# coding: utf-8

# In[121]:

import cv2, cPickle, os, glob, math, datetime, time
import numpy as np

def get_im(path):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (128, 96))
    return resized

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(3):  #this would have been 10
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for i,fl in enumerate(files):
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)
    return X_train, y_train

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        cPickle.dump(data, file,2)  #use cPickle and protocol_version:2
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = cPickle.load(file)
    return data

cache_path = os.path.join('mycnn_cache', 'train.dat')
if not os.path.isfile(cache_path):
    train_data, train_target = load_train()
    print 'Caching ...'; t1 = time.time()
    cache_data((train_data, train_target), cache_path)
    print 'Cached ... ',time.time() - t1,'s' 
else:
    print('Restore train from cache!'); t1 = time.time()
    (train_data, train_target) = restore_data(cache_path)
    print 'Restored ... ', time.time() - t1,'s'


# In[51]:

batch_size = 128
nb_classes = 10
nb_epoch = 1
img_rows, img_cols = 96, 128
nb_filters = 32
nb_pool = 2
nb_conv = 3

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from sklearn.cross_validation import train_test_split


# In[128]:

# index = T.dscalar()  # index to a [mini]batch
x = T.tensor4('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
temp_batch = T.dscalar('temp_batch')

(train_data, train_target) = restore_data(cache_path)
train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)
train_data /= 255
train_data = train_data.astype('float32')
print train_data.shape
# train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
print '[SHAPE][train]:',train_data.shape

nkerns = [32,64]
from theano.tensor.signal import downsample

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, image_shape, filter_shape, poolsize):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /  np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX
                ), borrow=True)
        b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv.conv2d(input=input, filters=self.W,
                    filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
#         self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = theano.tensor.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
    
    def return_output(self):
        return self.output

srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
def drop(input, p=0.5, rng=rng): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
    """            
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mas
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=theano.tensor.nnet.relu, p=0.5):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
#         train_output = drop(np.cast[theano.config.floatX](1./p) * output)
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
#         self.output = T.switch(T.neq(is_train, 0), train_output, output)
        
        # parameters of the model
        self.params = [self.W, self.b]

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    def predict(self):
        return self.y_pred


# In[129]:

rng = np.random.RandomState(1234)
# layer0_input = x.reshape((batch_size, 1, 96, 128))
layer0_input = x
# print temp_batch, type(temp_batch), type(x)
#(128, 1, 96, 128) -> (128, 32, 94, 126) -> (128 ,32, 47, 63)
layer0 = LeNetConvPoolLayer(rng, layer0_input, (batch_size, 1, img_rows, img_cols), (32, 1, nb_conv, nb_conv), (2,2))

#here rows = (96 - 3 + 1)/2; cols = (128 - 3 + 1)/2 ----> (128,32,47,63) --> (128,32,45,61) --> (128, 32, 22, 30)
layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 47, 63), 
                            filter_shape=(nkerns[0], nkerns[0], nb_conv, nb_conv), poolsize=(2, 2))

print 'Layer1 O/p:', layer1.output.shape
layer2_input = layer1.output.flatten(2)   #convert to two dimnesions
print 'Layer2:',layer2_input.shape

layer2 = HiddenLayer(rng, input=layer2_input, n_in= 32 * 22 * 30, n_out=10, activation=T.tanh)

layer3 = LogisticRegression(input=layer2.output, n_in=10, n_out=10)
cost = layer3.negative_log_likelihood(y)

validate_model = theano.function([x,y], layer3.errors(y))

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)
updates = []; learning_rate=0.025;
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))

train_model = theano.function([x,y], cost, updates=updates)


# In[130]:

from collections import Counter
X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_target, test_size=0.2)
print 'Crunching .......(Train):',len(X_train), '-',Counter(Y_train),' (Test):',len(X_test),'-',Counter(Y_test)
n_epochs = 10; epoch = 0;
while (epoch < n_epochs):
    epoch = epoch + 1
    for minibatch_index in xrange(len(X_train)/batch_size):
        iter = (epoch - 1) * len(X_train)/batch_size + minibatch_index
#         print '-------------------------Epoch:',epoch,' Iter:', iter, ' MiniIdx:',minibatch_index
        if iter % 10 == 0:
            print '-------------------------Epoch:',epoch,' training@iter = ', iter
        temp_X = np.array(X_train[int(minibatch_index) * batch_size: (int(minibatch_index) + 1) * batch_size], dtype='float32')
        temp_Y = np.array(Y_train[int(minibatch_index) * batch_size: (int(minibatch_index) + 1) * batch_size], dtype='int32')
        if len(temp_X) == batch_size:
            temp_X = temp_X.reshape(batch_size, 1, 96,128)
            cost_ij = train_model(temp_X, 
                          temp_Y)
    validation_losses = []
    for val_batch_idx in xrange(len(X_test)/batch_size):
        temp_X = np.array(X_test[val_batch_idx*batch_size: (val_batch_idx+1)*batch_size], dtype='float32')
        temp_Y = np.array(Y_test[val_batch_idx*batch_size: (val_batch_idx+1)*batch_size], dtype='int32')
        if len(temp_X) == batch_size:
            temp_X = temp_X.reshape(batch_size, 1, 96,128)
            validation_losses.append(validate_model(temp_X, temp_Y))
    print 'Epoch:',epoch, ' Loss:',np.mean(validation_losses)

print 'Done With:'  


