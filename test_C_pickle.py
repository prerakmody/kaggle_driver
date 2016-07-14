import cPickle
import pickle
import cv2
import time, os, glob, math

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 96))
    return resized

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
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
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        cPickle.dump(data, file,2)
        # pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = cPickle.load(file)
    return data

# cache_path = os.path.join('cache_exp', 'train.dat')
# if not os.path.isfile(cache_path):
# 	print 'Creating dat file...'
# 	t1 = time.time()
# 	train_data, train_target = load_train()
# 	print 'Loading time:',time.time() - t1,'s'
# 	t1 = time.time()
# 	cache_data((train_data, train_target), cache_path)
# 	print 'Writing time:',time.time() - t1,'s'
# else:
#     print('Restore train from cache!')
#     t1 = time.time()
#     (train_data, train_target) = restore_data(cache_path)
#     print time.time() - t1,'s'
#     print train_data[0], train_data[:10]

cache_path = os.path.join('cache_exp', 'test.dat')
if not os.path.isfile(cache_path):
	print 'Reading test data ...'
	t1 = time.time()
	test_data, test_id = load_test()
	print 'Time to read:', time.time() - t1,'s'
	print 'Writing to file'
	t1 = time.time()
	cache_data((test_data, test_id), cache_path)
	print 'Time to write:',time.time() - t1
else:
    print('Restore test from cache!')
    t1 = time.time()
    (test_data, test_id) = restore_data(cache_path)
    print 'Time taken to restore:', time.time() - t1,'s'
    print len(test_data)
    print test_data[:10]