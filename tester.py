#########################################################################
# File Name: tester.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-6-18 18:17
# Description: This script is used to implement SCWS based word
#   vector evaluation.
#########################################################################
from gensim.models import KeyedVectors
import numpy as np
from spearman import spearman, scws
import argparse


# Filter out those lines with no context words in vocab.
def filter_context(c, model):
    ret = list()
    for it in c:
        if it in model:
            ret.append(it)
    return ret


# Cosine Function
def cosine(v1, v2):
    return sum(v1*v2)/(sum(v1**2)**0.5)/(sum(v2**2)**0.5)


# Return the probability of each sense given a word and its context vector
# The return values are normalized to a probability distribution
def prob(glbmodel, ctxmodel, w, c):
    context = np.zeros(len(glbmodel['dog']))
    cnt = 0
    for it in c:
        if it in glbmodel:
            context += glbmodel[it]
            cnt += 1
    if cnt > 0:
        context = context / cnt
    p = list()
    for i in range(100):
        if w+'_s'+str(i) in ctxmodel:
            p.append(1+cosine(context, ctxmodel[w+'_s'+str(i)]))
    sp = sum(p)
    p = [x/sp for x in p]
    return p


# Return the local similarity between two words given context
def similarity_local(glbmodel, locmodel, ctxmodel, w1, w2, c1, c2):
    p1 = prob(glbmodel, ctxmodel, w1, c1)
    p2 = prob(glbmodel, ctxmodel, w2, c2)
    pos1 = np.argmax(p1)
    pos2 = np.argmax(p2)
    return locmodel.similarity(w1+'_s'+str(pos1),w2+'_s'+str(pos2))


# Find an element in a list
def find(l, element):
    for i in range(len(l)):
        if l[i] == element:
            return i
    return -1


# Return the test words and context list from WSI 2007 task
def sense_induction_test_word_with_context(idx, sentence):
    dt = 5
    words = sentence.split()
    pos = find(words, '<head>') + 1
    word = idx.split('.')[0]
    words = words[:pos-1] + [words[pos]] + words[pos+2:]
    context_list = list()
    for i in range(max(0, pos-dt), pos):
        context_list.append(words[i])
    for i in range(pos+1, min(pos+dt+2, len(words))):
        context_list.append(words[i])
    return word, context_list


# Given a test line in SCWS data set, output the context vector
def test_words_with_context(words):
    dt = 5
    c1 = words.split('\t')[5].split()
    c2 = words.split('\t')[6].split()
    ret = []
    words = words.split()
    for i in range(len(words)):
        if words[i] == '<b>':
            ret.append(words[i+1].lower())
    ret.append([])
    ret.append([])
    for i in range(len(c1)):
        if c1[i] == '<b>':
            for j in range(max(0,i-dt),i):
                ret[2].append(c1[j].lower())
            for j in range(i+3,min(i+3+dt,len(c1))):
                ret[2].append(c1[j].lower())
    for i in range(len(c2)):
        if c2[i] == '<b>':
            for j in range(max(0,i-dt),i):
                ret[3].append(c2[j].lower())
            for j in range(i+3,min(i+3+dt,len(c2))):
                ret[3].append(c2[j].lower())
    return ret


# Test a given model with global, local and context vector
def test_model(glbmodel, locmodel, ctxmodel, test_path):
    err_cnt = 0
    res = list()
    for line in open(test_path):
        [w1, w2, c1, c2] = test_words_with_context(line.lower())
        c1 = filter_context(c1, glbmodel)
        c2 = filter_context(c2, glbmodel)
        if (w1 not in glbmodel) or (w2 not in glbmodel) or len(c1) == 0 or len(c2) == 0:
            err_cnt += 1
            res.append(-1)
        else:
            res.append(similarity_local(glbmodel, locmodel, ctxmodel, w1, w2, c1, c2))
    return err_cnt, spearman(res, scws(test_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for pseudo multi-sense elimination.')
    parser.add_argument('--global-model', type=str, default='',
                        help='global vector path for evaluation')
    parser.add_argument('--context-model', type=str, default='',
                        help='context vector path for evaluation')
    parser.add_argument('--local-model', type=str, default='',
                        help='local vector path for evaluation')
    parser.add_argument('--test-path', type=str, default='../../data/SCWS/ratings.txt',
                        help='test path of SCWS data set')
    args = parser.parse_args()
    if args.context_model == '':
        raise Exception('Must identify the context model.')
    if args.local_model == '':
        raise Exception('Must identify the local model.')
    if args.global_model == '':
        raise Exception('Must identify the global model.')

    global_model = KeyedVectors.load_word2vec_format(args.global_model)
    context_model = KeyedVectors.load_word2vec_format(args.context_model)
    local_model = KeyedVectors.load_word2vec_format(args.local_model)
    error_count, score = test_model(global_model, local_model, context_model, args.test_path)
    print error_count, score * 100
