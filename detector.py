#########################################################################
# File Name: detector.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-6-18 18:17
# Description: This script is used to implement unsupervised pseudo
#   multi-sense detector.
#########################################################################
from gensim.models import KeyedVectors
from pprint import pprint


class PsdMulDetector:
    def __init__(self, global_model, local_model):
        self.locmodel = local_model
        self.glbmodel = global_model
        self.passed = set()

    '''
    Find nearest neighbors for a word in a specific model.
    w : the word
    k : the number when finding nearest neighbors (default: 20)
    '''
    @staticmethod
    def nearest_neighbors(model, w, k=20):
        return model.most_similar(positive=[w], topn=k)

    '''
    Returns the prototype of a word in local model.
    w : the word
    '''
    @staticmethod
    def prototype(w):
        return w[:w.rfind('_s')]

    '''
    Get the same word set from local model given a proto.
    proto : the given proto
    '''
    @staticmethod
    def get_same_word_list(model, proto):
        word_list = list()
        suffix = 0
        while True:
            if proto+('_s%d' % suffix) in model:
                word_list.append(proto+('_s%d' % suffix))
                suffix += 1
            else:
                break
        return word_list

    '''
    Detect pseudo multi-sense based on nearest neighbors and global vectors.
    '''
    def detect(self):
        print 'Begin to detect pseudo multi-sense cases.'
        res = list()
        for w in self.locmodel.vocab:
            proto = self.prototype(w)
            if proto+'_s1' not in self.locmodel:
                self.passed.add(proto)
                print 'detect', len(self.passed)
                continue
            if proto in self.passed:
                continue
            else:
                self.passed.add(proto)
                same_word = self.get_same_word_list(self.locmodel, proto)
                nearest_neighbors = map(
                    lambda x: self.nearest_neighbors(self.locmodel, x),
                    same_word
                )
                for j in range(len(nearest_neighbors)):
                    for i in range(j):
                        similarity = 0
                        for nn_i in nearest_neighbors[i]:
                            for nn_j in nearest_neighbors[j]:
                                similarity += self.glbmodel.similarity(
                                    self.prototype(nn_i[0]), self.prototype(nn_j[0]))
                        res.append((same_word[i], same_word[j], similarity))
        res = sorted(res, key=lambda x: -x[2])
        max_value = res[0][2]
        res = map(lambda x: (x[0], x[1], x[2]/max_value), res)
        return res

    '''
    Combine the detected pseudo multi-sense cases.
    '''
    @staticmethod
    def combine(seeds):
        temp = dict()
        for item in seeds:
            if item[0] not in temp:
                temp[item[0]] = set()
            if item[1] not in temp:
                temp[item[1]] = set()
            temp[item[0]].add(item[0])
            temp[item[0]].add(item[1])
            temp[item[1]].add(item[0])
            temp[item[1]].add(item[1])
        for item in temp:
            for elem1 in temp[item]:
                for elem2 in temp[item]:
                    temp[elem1].add(elem2)
                    temp[elem2].add(elem1)
        ret = list()
        for item in temp:
            elems = sorted(list(temp[item]))
            if item == elems[0]:
                ret.append(elems)
        return ret

if __name__ == '__main__':
    glbmodel = KeyedVectors.load_word2vec_format(
        '../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_global.vec'
    )
    locmodel = KeyedVectors.load_word2vec_format(
        '../../data/vectors/neelakantan_vectors.NP-MSSG.50D.6K_local.vec'
    )
    detector = PsdMulDetector(glbmodel, locmodel)
    result = detector.detect()
    with open('debug.log', 'w') as fout:
        pprint(result, fout)
