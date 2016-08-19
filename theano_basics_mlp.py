#https://gist.github.com/honnibal/6a9e5ef2921c0214eeeb
import theano
import theano.tensor as T
import numpy as np
import os, sys, time, gzip, cPickle, numpy
from os import path

def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        data_dir = os.path.join(os.path.split(__file__)[0], "data")
        if not path.exists(data_dir):
            print "No data directory to save data to. Try:"
            print "mkdir ../data"
            sys.exit(1)
        new_path = path.join(data_dir, data_file)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % url
        urllib.urlretrieve(url, dataset)

    print '... loading data'

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return _make_array(train_set), _make_array(valid_set), _make_array(test_set)


def _make_array(xy):
    data_x, data_y = xy
    return zip(
        numpy.asarray(data_x, dtype=theano.config.floatX),
        numpy.asarray(data_y, dtype='int32'))

def _init_logreg_weights(n_hidden, n_out):
	weights = np.zeros((n_hidden, n_out), dtype=theano.config.floatX)
	bias = np.zeros((n_out,), dtype=theano.config.floatX)
	return (
		theano.shared(name='W', borrow=True, value=weights),
		theano.shared(name='b', borrow=True, value=bias)
		)

def _init_hidden_weights(n_in, n_out):
	rng = np.random.RandomState(1234)
	weights = np.asarray(
			rng.uniform(low=6/(n_in+n_out), high=6/(n_in+n_out), size=(n_in, n_out)),
			dtype = theano.config.floatX
		)
	bias = np.zeros((n_out,), dtype=theano.config.floatX)
	return (
		theano.shared(name='W', value=weights, borrow=True),
		theano.shared(name='n', value=bias, borrow=True)
		)

def feed_forward(activation, weights, bias, input_):
	return activation(T.dot(input_, weights) + bias)

def sgd_step(param, cost, learning_rate):
	return param - learning_rate * T.grad(cost, wrt = param)

def L1(L1_reg, w1, w2):
	return L1_reg * (abs(w1).sum() + abs(w2).sum())

def L2(L2_reg, w1, w2):
	return L2_reg * (abs(w1).sum() + abs(w2).sum())

def compile_model(n_in, n_classes, n_hidden, learning_rate, L1_reg, L2_reg):
	x = T.vector('x')
	y = T.iscalar('y')

	hidden_w, hidden_b = _init_hidden_weights(n_in, n_hidden)
	logreg_w, logreg_b = _init_logreg_weights(n_hidden, n_classes)

	p_y_given_x = feed_forward(T.nnet.softmax, logreg_w, logreg_b, feed_forward(T.tanh, hidden_w, hidden_b, x))

	cost = (
		-T.log(p_y_given_x[0, y]) + L1(L1_reg, logreg_w, hidden_w) + L2(L2_reg, logreg_w, hidden_w)
		)

	train_model = theano.function(
			inputs = [x,y],
			outputs = [cost],
			updates = [
				(logreg_w, sgd_step(logreg_w, cost, learning_rate)),
				(logreg_b, sgd_step(logreg_b, cost, learning_rate)),
				(hidden_w, sgd_step(hidden_w, cost, learning_rate)),
				(hidden_b, sgd_step(hidden_b, cost, learning_rate))
			]
		)

	evaluate_model = theano.function(
		inputs = [x,y],
		outputs = [T.neq(y, T.argmax(p_y_given_x[0]))]
		)
	return train_model, evaluate_model

def main(learning_rate = 0.01, L1_reg = 0.0, L2_reg = 0.001, n_epochs=1000, dataset = 'mnist.pkl.gz', n_hidden = 500):
	train_examples, dev_examples, test_examples = load_data(dataset)
	print '... building the model'
	train_model, evaluate_model = compile_model(28*28, 10, n_hidden, learning_rate, L1_reg, L2_reg)
	for epoch in range(1, n_epochs+1):
		for x,y in train_examples:
			train_model(x,y)
		# compute zero-one loss on validation set
		error = np.mean([evaluate_model(x,y) for x,y in dev_examples])
		print('epoch %i, validation error %f %%' % (epoch, error * 100))

if __name__ == "__main__":
	main()