import numpy as np
np.random.seed(2016)
import os, glob, math, datetime, time
import cv2
import cPickle as pickle
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.utils import np_utils

w, h = 128, 96

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))
    return resized

def expand_train(img):
    res = []
    new_img = np.roll(img, -1, 0); new_img[h-1,:] = np.zeros(w)   #HOR UP
    res.append(new_img)
    new_img = np.roll(img, 1, 0); new_img[0,:] = np.zeros(w)      #HOR Down
    res.append(new_img)
    new_img = np.roll(img, -1, 1); new_img[:,w-1] = np.zeros(h)   #VER LEFT
    res.append(new_img)
    new_img = np.roll(img, 1, 1);  new_img[:,0] = np.zeros(h)     #VER RIGHT
    res.append(new_img)
    return res

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(1):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for i,fl in enumerate(files):
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)
            #every nth image, expand the size of the dataset
            if i % 20 == 0:
                X_expand = expand_train(img)
                for each in X_expand:
                    X_train.append(each)
                    y_train.append(j)
    return X_train, y_train

def load_test():
    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for i,fl in enumerate(files):
        if i%10 != 0: continue
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%1000 == 0:
            print('i:',i, 'images:',len(total), 'Files:', len(files))

    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file,2)  #use cPickle and protocol_version:2
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def create_submission(predictions, test_id, loss):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) 
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    print '=========================',sub_file,'================================='
    result1.to_csv(sub_file, index=False)


# The same as log_loss
def mlogloss(target, pred):
    score = 0.0
    for i in range(len(pred)):
        pp = pred[i]
        for j in range(len(pp)):
            prob = pp[j]
            if prob < 1e-15:
                prob = 1e-15
            score += target[i][j] * math.log(prob)
    return -score/len(pred)


def validate_holdout(model, holdout, target):
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)
    # score = model.evaluate(holdout, target, show_accuracy=True, verbose=0)
    # print('Score holdout: ', score)
    # score = mlogloss(target, predictions)
    # print('Score : mlogloss', score)
    return score


cache_path = os.path.join('cache', 'train.dat')
print cache_path, os.path.isfile(cache_path) 
if not os.path.isfile(cache_path):
    train_data, train_target = load_train()
    print 'Caching ...'; t1 = time.time()
    cache_data((train_data, train_target), cache_path)
    print 'Cached ... ',time.time() - t1,'s' 
else:
    print('Restore train from cache!'); t1 = time.time()
    (train_data, train_target) = restore_data(cache_path)
    print 'Restored ... ', time.time() - t1,'s'

batch_size = 64    #[PONDER]can this be made bigger??
nb_classes = 10
nb_epoch = 6
# input image dimensions
img_rows, img_cols = 96, 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)
train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
train_data = train_data.astype('float32')
train_data /= 255
print '[SHAPE][train]:',train_data.shape
train_target = np_utils.to_categorical(train_target, nb_classes)
print '[SHAPE][train][target]:',train_target.shape

X_train, X_test, X_holdout, Y_train, Y_test, Y_holdout = split_validation_set_with_hold_out(train_data, train_target, 0.2)
print('Split train: ', len(X_train))
print('Split valid: ', len(X_test))
print('Split holdout: ', len(X_holdout))


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


model_from_cache = 0
if model_from_cache == 1:
    model = read_model()
else:
    t1 = time.time()
    print '======================================DEEP LEARNING==================================================='
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(1, img_rows, img_cols)))
    model.add(BatchNormalization())  #this kinda messes up with the model in some way
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', init='he_normal'))
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    print 'Model Compilation:',time.time()-t1,'s'
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
              validation_data=(X_test, Y_test))
    print 'Model Fitted ...', time.time() - t1,'s'

score = model.evaluate(X_test, Y_test, verbose=0)
print('Score: ', score, ' ', time.time() - t1,'s')
score = model.evaluate(X_holdout, Y_holdout, verbose=0)
print('Score holdout: ', score,' ', time.time() - t1,'s')
validate_holdout(model, X_holdout, Y_holdout)
save_model(model)


cache_path = os.path.join('cache', 'test.dat')
if not os.path.isfile(cache_path):
    test_data, test_id = load_test()
    cache_data((test_data, test_id), cache_path)
else:
    print('Restore test from cache!'); t1 = time.time()
    (test_data, test_id) = restore_data(cache_path)
    print 'Restored test data ... ', time.time() - t1,'s'

test_data = np.array(test_data, dtype=np.uint8)
test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
test_data = test_data.astype('float32')
test_data /= 255
print('Test shape:', test_data.shape)
predictions = model.predict(test_data, batch_size=128, verbose=1)

create_submission(predictions, test_id, score[0])

"""

# """
# >>> from theano.sandbox.cuda.dnn import *
# Using gpu device 0: GeForce 840M (CNMeM is disabled, cuDNN not available)
# >>> print(dnn_available())
# False
# >>> print(dnn_available.msg)
# Can not compile with cuDNN. We got this error:
# """