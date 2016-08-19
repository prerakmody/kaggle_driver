import cv2, cPickle, os, glob, math, datetime, time
import numpy as np
from collections import Counter

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

class CNN(object):
    def __init__(train_data, test_data=[], rng, srng, batch_size = 128, nb_classes = 10, nb_epoch = 10, img_rows = 96,  img_cols = 128, 
            nb_filters = [32,64], nb_pool = 2, nb_conv = 3, init_learning_rate=0.025, validate_size = 0.3):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nb_filters = nb_filters
        self.nb_pool = nb_pool
        self.nb_conv = nb_conv
        self.rng = rng
        self.srng = srng
        self.init_learning_rate = init_learning_rate

        self.train_data = np.array(train_data, dtype=np.uint8)
        self.train_target = np.array(train_target, dtype=np.uint8)
        print '[DEBUG]:', self.train_data[0]
        self.train_data /= 255
        self.train_data = train_data.astype('float32')
        print '[DEBUG]:', self.train_data[0]
        self.validate_size = validate_size

        self.x = T.tensor4('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels
        is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
        

    def createCNN(self):
        layer0_input = self.x.reshape((self.batch_size, 1, self.img_rows, self.img_cols))
        #(128, 1, 96, 128) -> (128, 32, 94, 126) -> (128 ,32, 47, 63)
        print 'Layer0:',self.batch_size, 1, self.img_rows, self.img_cols
        layer0 = ConvPoolLayer(self.rng, layer0_input, (self.batch_size, 1, self.img_rows, self.img_cols), 
            (self.batch_size, 1, self.nb_conv, self.nb_conv), (self.nb_pool, slf.nb_pool))

        #(128, 32, 47, 63) --> (128, 32, 45, 61) --> (128, 32, 22, 30)
        new_rows = int((self.img_rows - self.nb_conv + 1)/2) #since stride is 1
        new_cols = int((self.img_cols -self.nb_conv + 1)/2)
        print 'Layer1:', self.batch_size, self.nb_filters[0], new_rows, new_cols
        layer1 = ConvPoolLayer(self.rng, input=layer0.output, image_shape=(self.batch_size, self.nb_filters[0], new_rows, new_cols), 
                            filter_shape=(nb_filters[0], nb_filters[0], nb_conv, nb_conv), poolsize=(self.nb_pool, slf.nb_pool))

        # (128, 32, 22, 30) -> (128, 32, 20, 28) -> (128, 64, 10, 14)
        new_rows = int((new_rows - self.nb_conv + 1)/2)
        new_cols = int((new_cols - self.nb_conv + 1)/2)
        print 'Layer2:', self.batch_size, self.nb_filters[0], new_rows, new_cols
        layer2 = ConvPoolLayer(self.rng, input=layer1.output, image_shape = (self.batch_size, self.nb_filters[0], new_rows, new_cols),
                           filter_shape=(self.nb_filters[1], self.nb_filters[0], nb_conv, nb_conv), poolsize=(self.nb_pool, slf.nb_pool))

        # (64, 10, 14) -> (64,10,14)
        new_rows = int((new_rows - self.nb_conv + 1)/2)
        new_cols = int((new_cols - self.nb_conv + 1)/2)
        layer3_input = layer2.output.flatten(2)   #convert to two dimnesions
        layer3 = HiddenLayer(rng, input=layer3_input, n_in= self.nb_filters[1] * new_rows * new_cols, n_out = self.nb_filters[1] * new_rows * new_cols, 
            is_train=is_train,activation=theano.tensor.nnet.relu)        

        #(64, 10, 14) -> (10, 14)
        layer4 = HiddenLayer(rng, input=layer3.output, n_in= self.nb_filters[1] * new_rows * new_cols, n_out = new_rows * new_cols, 
            is_train=is_train,activation=theano.tensor.nnet.relu)

        layer5 = LogisticRegression(input=layer4.output, n_in = new_rows * new_cols, n_out = nb_classes)

        self.test_model = theano.function([x], layer5.p_y_given_x)

        cost = layer5.negative_log_likelihood(y)

        self.validate_model = theano.function([x,y,is_train], layer5.errors(y))

        params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)
        updates = [];
        learning_rate = theano.shared(np.cast[theano.config.floatX](self.init_learning_rate) )
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        self.train_model = theano.function([x,y, is_train], cost, updates=updates)

    def train(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.train_data, self.train_target, test_size = self.validate_size)
        print 'Crunching .......(Train):',len(X_train), '-',Counter(Y_train),' (Test):',len(X_test),'-',Counter(Y_test)
        epoch = 0;
        while (epoch < self.nb_epoch):
            epoch = epoch + 1
            print "learning rate: ", self.learning_rate.get_value() 
            for minibatch_index in xrange(len(X_train)/self.batch_size):
                iter = (epoch - 1) * len(X_train)/self.batch_size + minibatch_index
                if iter % 10 == 0:
                    print '-------------------------Epoch:',epoch,' training@iter = ', iter
                temp_X = np.array(X_train[int(minibatch_index) * self.batch_size: (int(minibatch_index) + 1) * self.batch_size], dtype='float32')
                temp_Y = np.array(Y_train[int(minibatch_index) * self.batch_size: (int(minibatch_index) + 1) * self.batch_size], dtype='int32')
                if len(temp_X) == self.batch_size:
                    temp_X = temp_X.reshape(self.batch_size, 1, self.img_rows, self.img_cols)  #very.very.important.step
                    cost_ij = self.train_model(temp_X, temp_Y,
                                         np.cast['int32'](1))
            validation_losses = []
            for val_batch_idx in xrange(len(X_test)/self.batch_size):
                temp_X = np.array(X_test[val_batch_idx*self.batch_size: (val_batch_idx+1)*self.batch_size], dtype='float32')
                temp_Y = np.array(Y_test[val_batch_idx*self.batch_size: (val_batch_idx+1)*self.batch_size], dtype='int32')
                if len(temp_X) == self.batch_size:
                    temp_X = temp_X.reshape(self.batch_size, 1, self.img_rows, self.img_cols)
                    validation_losses.append(self.validate_model(temp_X, temp_Y,
                                                           np.cast['int32'](0)))
            print 'Epoch:',epoch, ' Loss:',np.mean(validation_losses)

        print 'Done With:' 

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

class HiddenLayer(object):
    def __init__(self, rng, srng, input, n_in, n_out, is_train, W=None, b=None, activation=theano.tensor.nnet.relu, p=0.5):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform( low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            # if activation == theano.tensor.nnet.sigmoid:
            #     W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = self.drop(np.cast[theano.config.floatX](1./p) * self.output, p, rng=srng)
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, self.output)
        
        # parameters of the model
        self.params = [self.W, self.b]

    def drop(self, input, p, srng): 
        mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

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

    def negative_log_likelihood(self, y):
        return -y*T.log(self.p_y_given_x) – (1-y)*T.log(1–self.p_y_given_x)

    def predict(self):
        return self.y_pred

if __name__ == "__main__":
    train_data, train_target = getImgData('train.dat', 'train',no_classes=2).get_data()
    test_data, test_target = getImgData('test.dat', 'test').get_data()

    import theano
    import theano.tensor as T
    from theano.tensor.signal import pool
    from theano.tensor.nnet import conv
    from theano.tensor.signal import downsample
    from sklearn.cross_validation import train_test_split
    rng = numpy.random.RandomState(23455)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    # CNN(train_data, test_data, rng, srng).createCNN()
    CNN(train_data, rng, srng).createCNN()
    
    

"""

Conv Layer0: 64 1 96 128
Conv Layer1: 64 32 47 63
Conv Layer2: 64 32 22 30
Hidden Layer3: 64 10 14 ( 8960 )

Conv Layer0: 128 1 96 128
Conv Layer1: 128 32 47 63
Conv Layer2: 128 32 22 30
Hidden Layer3: 64 10 14 ( 8960 )
Hidden layer4: 64 10 14 ( 8960 )
Logistic layer5: 10 14 ( 140 )


#getting the output in terms of probabilities of each class
import pandas as pd
df = pd.Dataframe()

cache_path = os.path.join('cache_exp', 'test.dat')
print('Restore train from cache!'); t1 = time.time()
(test_data, test_target) = restore_data(cache_path)
print 'Restored ... ', time.time() - t1,'s'

"""
