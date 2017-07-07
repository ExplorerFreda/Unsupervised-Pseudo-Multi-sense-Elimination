#########################################################################
# File Name: eliminator.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-6-18 18:17
# Description: This script is used to implement unsupervised pseudo
#   multi-sense eliminator.
#########################################################################
import random
import torch
from torch.autograd import Variable
from gensim.models import KeyedVectors
from detector import PsdMulDetector
import argparse
import pprint
import codecs


class MultiSenseVectorSpace:
    def __init__(self, global_model, local_model, context_model):
        self.glbmodel = global_model
        self.locmodel = local_model
        self.ctxmodel = context_model
        self.detector = PsdMulDetector(self.glbmodel, self.locmodel)
        self.dim = len(self.locmodel['dog_s0'])
        self.transform_matrix = Variable(torch.randn(self.dim, self.dim), requires_grad=True)

    '''
    Train the pseudo multi-sense eliminating matrix.
    theta: the threshold to determine whether a pair of senses is pseudo multi-sense
    alpha: learning rate
    '''
    def train_matrix(self, theta=0.5, alpha=0.2):
        print 'Begin to train matrix.'
        seeds = filter(lambda x: x[2] > theta, self.detector.detect())
        flag = True
        last = 1e8
        while flag:
            random.shuffle(seeds)
            loss = 0
            for item in seeds:
                x0 = Variable(torch.Tensor(self.locmodel[item[0]])).resize(self.dim, 1)
                x1 = Variable(torch.Tensor(self.locmodel[item[1]])).resize(self.dim, 1)
                y = (x0 + x1) * 0.5
                res = self.transform_matrix.mm(x0) - y
                loss += res.t().mm(res)
                res = self.transform_matrix.mm(x1) - y
                loss += res.t().mm(res)
            loss = loss / len(seeds) / 2
            loss.backward()
            self.transform_matrix -= self.transform_matrix.grad * alpha
            self.transform_matrix = Variable(self.transform_matrix.data, requires_grad=True)
            print loss.data[0][0]
            if 0 < last - loss.data[0][0] < 0.001:
                break
            elif last < loss.data[0][0]:
                alpha *= 0.1
            last = loss.data[0][0]

    '''train matrix with the compressed method'''
    def train_matrix_compress(self, theta=0.5, alpha=0.2):
        print 'Begin to train matrix.'
        seeds = PsdMulDetector.combine(filter(lambda x: x[2] > theta, self.detector.detect()))
        flag = True
        last = 1e8
        while flag:
            total_seed_items = 0
            random.shuffle(seeds)
            loss = 0
            for item in seeds:
                center = 0
                for x in item:
                    center += Variable(torch.Tensor(self.locmodel[x])).resize(self.dim, 1)
                center = center / len(item)
                for x in item:
                    res = self.transform_matrix.mm(
                        Variable(torch.Tensor(self.locmodel[x])).resize(self.dim, 1)
                    ) - center
                    loss += res.t().mm(res)
                total_seed_items += len(item)
            loss = loss / total_seed_items
            loss.backward()
            self.transform_matrix -= self.transform_matrix.grad * alpha
            self.transform_matrix = Variable(self.transform_matrix.data, requires_grad=True)
            print loss.data[0][0]
            if 0 < last - loss.data[0][0] < 0.001:
                break
            elif last < loss.data[0][0]:
                alpha *= 0.1
            last = loss.data[0][0]

    '''
    Store the transformed space into other files.
    '''
    def store(self, output_prefix):
        print 'Begin to store matrix into a file'
        local_filename = output_prefix + '.vec'
        with codecs.open(local_filename, 'w', 'utf8') as fout:
            fout.write('%d %d\n' % (len(self.locmodel.vocab), self.dim))
            for word in self.locmodel.vocab:
                x = self.transform_matrix.mm(
                    Variable(torch.Tensor(self.locmodel[word])).resize(self.dim, 1))
                fout.write(word)
                for i in range(self.dim):
                    fout.write(' ' + str(x.data[i][0]))
                fout.write('\n')
        fout.close()

        matrix_filename = output_prefix + '.mat'
        with open(matrix_filename, 'w') as fout:
            for item in self.transform_matrix.data:
                for p in item:
                    fout.write(str(p) + ' ')
                fout.write('\n')
        fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for pseudo multi-sense elimination.')
    parser.add_argument('--global-model', type=str, default='',
                        help='global vector path for evaluation')
    parser.add_argument('--context-model', type=str, default='',
                        help='context vector path for evaluation')
    parser.add_argument('--local-model', type=str, default='',
                        help='local vector path for evaluation')
    parser.add_argument('--store-path', type=str, default='',
                        help='the path to store the transformed vector, without .vec suffix')
    parser.add_argument('--pseudo-threshold', type=float, default=0.5,
                        help='the threshold for pseudo multi-sense detection')
    parser.add_argument('--learning-rate', type=float, default=0.2,
                        help='the initial learning rate')
    parser.add_argument('--training-method', type=str, default='pairwise',
                        help='pairwise or compress, indicates how to train matrix in the elimination procedure')
    args = parser.parse_args()
    if args.context_model == '':
        raise Exception('Must identify the context model.')
    if args.local_model == '':
        raise Exception('Must identify the local model.')
    if args.global_model == '':
        raise Exception('Must identify the global model.')
    if args.store_path == '':
        raise Exception('Must identify the store path.')

    glbmodel = KeyedVectors.load_word2vec_format(args.global_model)
    with codecs.open('debug.log', 'w', 'utf8') as fout:
        for item in glbmodel.vocab:
            fout.write(item+'\n')
        fout.close()

    locmodel = KeyedVectors.load_word2vec_format(args.local_model)
    ctxmodel = KeyedVectors.load_word2vec_format(args.context_model)
    space = MultiSenseVectorSpace(glbmodel, locmodel, ctxmodel)
    if args.training_method == 'pairwise':
        space.train_matrix(theta=args.pseudo_threshold, alpha=args.learning_rate)
    elif args.training_method == 'compress':
        space.train_matrix_compress(theta=args.pseudo_threshold, alpha=args.learning_rate)
    else:
        raise Exception('Training method must be pairwise or compress')
    space.store(args.store_path)
