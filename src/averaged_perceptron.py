'''
Created on 2017/03/12

@author: rindybell
'''

""" imports """
import sys
import os
import argparse
import numpy as np
from sklearn import metrics
from scipy import sparse
import scipy

""" variables """
""" functions """


class AveragedPerceptron:

    def __init__(self, input_dim, output_dim, max_iteration=10000, verbose=0):
        self.iter_weight = np.zeros((output_dim, input_dim))
        self.weight = np.zeros((output_dim, input_dim))
        self.max_iteration = max_iteration
        self.verbose = verbose
        # self.bias = np.zeros (output_dim)

    def fit(self, X, Y, valid_X, valid_Y):
        iters = 1
        # X = sparse.csr_matrix(X)
        # Y = sparse.csr_matrix(Y)

        for i in range(1, self.max_iteration):
            roop_end = True

            for (sample, gold) in zip(X, Y):

                output = np.matmul(self.iter_weight, sample)
                # output = scipy.dot(self.iter_weight, sample)
                # print output

                pred = np.argmax(output)
                if gold != pred:
                    roop_end = False

                    # gradient
                    u = sample

                    # update weight
                    self.iter_weight[gold] += u
                    self.iter_weight[pred] -= u

                    self.weight[gold] += (iters * u)
                    self.weight[pred] -= iters * u
                    iters += 1

            if self.verbose > 0:
                preds = self.predict(valid_X)
                print self.accuracy(preds, valid_Y)

            # is convergence ?
            if roop_end:
                break

        self.weight = self.iter_weight - (self.weight / iters)

    def accuracy(self, golds, preds):
        accuracy = metrics.accuracy_score(golds,
                                          preds)
        return accuracy

    def predict(self, X):
        preds = np.matmul(self.weight, X.transpose())
        return map(np.argmax, preds.transpose())

""" main """


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

    AP = AveragedPerceptron(
        input_dims, output_dims, max_iteration=1000, verbose=1)
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
