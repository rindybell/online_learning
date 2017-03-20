'''
Created on 2017/03/12

@author: rindybell
'''
import numpy as np
from scipy import sparse
import scipy
import argparse
from sklearn import metrics
from nbformat import current
from tensorflow.python.ops.nn_ops import top_k


class AROW:

    def __init__(self, input_dim, output_dim, r=0.1, max_iteration=100, top_k=2, verbose=0):
        self.weights = {}
        self.variances = {}
        for i in xrange(output_dim):
            self.weights[i] = np.zeros((input_dim, 1))
            self.variances[i] = np.eye(input_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.r = r
        self.max_iteration = max_iteration
        self.verbose = verbose
        self.top_k = 3

    def beta(self, x_t, x, variance):
        term = x_t.dot(variance).dot(x)[0][0] + self.r
        return 1.0 / (term)

    def alpha(self, y, x_t, weight, beta):
        term = (1.0 - y * (x_t.dot(weight))[0][0])
        # assert isinstance(term, int) == True
        return max(0, term) * beta

    def update(self, y, x, x_t, weight, variance):
        if (y * self.infer(x_t, weight)) < 1.0:
            beta = self.beta(x_t, x, variance)
            alpha = self.alpha(y, x_t, weight, beta)

            var_x = variance.dot(x)
            x_t_var = x_t.dot(variance)

            new_weight = weight + alpha * y * var_x
            # new_weight = weight + y * x
            new_variance = variance - beta * var_x.dot(x_t_var)

            return (new_weight, new_variance)
        else:
            return (weight, variance)

    def infer(self, x, weight):
        return x.dot(weight)[0]

    def prediction_sample(self, sample):
        label_scores = map(
            lambda x: (x[0], self.infer(sample, x[1])), self.weights.items())
        # print label_scores[:10]
        return sorted(label_scores, key=lambda x: x[1], reverse=True)[0][0]

    def fit(self, X, Y, valid_X, valid_Y, do_sparse=False):
        iters = 1

        for i in range(1, self.max_iteration):
            roop_end = True

            for (sample, gold) in zip(X, Y):
                x = np.reshape(sample, (self.input_dim, 1))
                x_t = x.transpose()

                top_k_tuples = map(
                    lambda x: (x[0], self.infer(sample, x[1])), self.weights.items())

                sorted_k_tuples = sorted(
                    top_k_tuples, key=lambda x: x[1], reverse=True)[:self.top_k]

                for (label, score) in sorted_k_tuples:
                    if gold == label:
                        y = 1
                    else:
                        y = -1

                    (new_weight, new_variance) = self.update(
                        y, x, x_t, self.weights[label], self.variances[label])

                    self.weights[label] = new_weight
                    self.variances[label] = new_variance

            if self.verbose != 0:
                predictions = self.predict(valid_X)
                print self.accuracy(valid_Y, predictions)

    def accuracy(self, golds, preds):
        accuracy = metrics.accuracy_score(golds,
                                          preds)
        return accuracy

    def predict(self, X):
        return map(lambda x: self.prediction_sample(x), X)


def main(options={}):
    from sklearn.datasets import load_digits
    digits = load_digits()
    (input_dims, output_dims) = (
        digits.data.shape[1], len(digits.target_names))

    train_X = digits.data[:1600]
    train_Y = digits.target[:1600]

    dev_X = digits.data[1600:1700]
    dev_Y = digits.target[1600:1700]

    test_X = digits.data[1700:]
    test_Y = digits.target[1700:]

    AP = AROW(
        input_dims, output_dims, max_iteration=10, verbose=1)
    AP.fit(train_X, train_Y, dev_X, dev_Y)
    print AP.accuracy(test_Y, AP.predict(test_X))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script is a template for any python scripts.")
    # parser.add_argument ("-v", "--var", type=str,
    #                      help=u"variable", default=None)
    args = parser.parse_args()

    options = args.__dict__

    if None in options.values():
        parser.print_help()
        exit(-1)
    main(options)
