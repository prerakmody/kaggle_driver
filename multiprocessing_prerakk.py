from multiprocessing import Pool
import pandas as pd
import os, random
import cv2
import time

results = []

def get_im(x):
    return cv2.resize(cv2.imread(x, 0), (128, 96)).flatten()

def log_result(retval):
    results.append(retval)
    if len(results) % 500 == 0:
        print len(results)

def dothis():
    dirr =  os.getcwd()
    labels = [i for i in os.listdir(os.path.join(dirr, 'imgs', 'train')) if 'c' in i]
    data = []
    for label in labels:
        paths = os.listdir(os.path.join(dirr, 'imgs','train', label))
        X = [(os.path.join(dirr, 'imgs', 'train', label, i), label) for i in paths]
        data.extend(X)
    random.shuffle(data) #why?
    df = pd.DataFrame({'paths':[i[0] for i in data], 'labels':[i[1] for i in data]})
    
    X = [each[0] for each in data]
    X_label = []
    for cl in labels: X_label.append(df['labels'] == cl)  #true/false
    # pool = Pool(4)
    t1 = time.time()
    for i,item in enumerate(X):
    	if i<10: print item
    	results.append(get_im(item))
    	if i%500 == 0: print len(results), '/', len(X)
    	# pool.apply_async(get_im, args=[item], callback=log_result)
    # pool.close() 
    # pool.join()
    print time.time() - t1,'s'
    print len(results)

if __name__ == "__main__":
	dothis()