#########################################################################
# File Name: corpus_tagger.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-06-22 16:25
# Description: This script is used to tag corpus with a self-paced
#   policy (controlled threshold by an external program).
#########################################################################

from gensim.models import KeyedVectors
from eliminator import MultiSenseVectorSpace
import argparse
import numpy as np
from tester import prob
from detector import PsdMulDetector
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corpus tagger.')
    parser.add_argument('--global-model', type=str, default='',
                        help='global vector path for evaluation')
    parser.add_argument('--context-model', type=str, default='',
                        help='context vector path for evaluation')
    parser.add_argument('--local-model', type=str, default='',
                        help='local vector path for evaluation')
    parser.add_argument('--input-corpus', type=str, default='',
                        help='input corpus')
    parser.add_argument('--output-corpus', type=str, default='',
                        help='output corpus')
    parser.add_argument('--tag-threshold', type=float, default=-1,
                        help='threshold for tagging')
    parser.add_argument('--pseudo-threshold', type=float, default=0.5,
                        help='threshold for pseudo multi-sense detection')
    parser.add_argument('--window', type=int, default=5,
                        help='window size for context when tagging corpus')
    args = parser.parse_args()
    if args.context_model == '':
        raise Exception('Must identify the context model.')
    if args.local_model == '':
        raise Exception('Must identify the local model.')
    if args.global_model == '':
        raise Exception('Must identify the global model.')
    if args.input_corpus == '':
        raise Exception('Must identify the input corpus path.')
    if args.output_corpus == '':
        raise Exception('Must identify the output corpus path.')

    glbmodel = KeyedVectors.load_word2vec_format(args.global_model)
    locmodel = KeyedVectors.load_word2vec_format(args.local_model)
    ctxmodel = KeyedVectors.load_word2vec_format(args.context_model)

    # similar pairs generation
    space = MultiSenseVectorSpace(glbmodel, locmodel, ctxmodel)
    pseudo_multi_sense_pairs = filter(lambda x: x[2] > args.pseudo_threshold, space.detector.detect())
    pseudo_multi_sense_seeds = PsdMulDetector.combine(pseudo_multi_sense_pairs)
    print pseudo_multi_sense_seeds
    fout = open('../../data/pseudo-multi-sense.json', 'w')
    fout.write(json.dumps(pseudo_multi_sense_seeds) + '\n')
    fout.close()
    exit()
    true_sense = dict()
    multisense_set = dict()
    for w in space.locmodel.vocab:
        if w[-1] != '1':
            continue
        multisense_set[PsdMulDetector.prototype(w)] = 0
    for cluster in pseudo_multi_sense_seeds:
        for item in cluster:
            proto = PsdMulDetector.prototype(item)
            assert proto in multisense_set
            true_sense[item] = proto + '_s' + str(multisense_set[proto])
        multisense_set[proto] += 1
    for w in space.locmodel.vocab:
        proto = PsdMulDetector.prototype(w)
        if (proto in multisense_set) and (w not in true_sense):
            true_sense[w] = proto + '_s' + str(multisense_set[proto])
            multisense_set[proto] += 1
    print len(multisense_set)
    print true_sense

    # output
    fout = open(args.output_corpus, 'w')
    for idx, line in enumerate(open(args.input_corpus, 'r')):
        words = line.split()
        for w in range(len(words)):
            if words[w] not in multisense_set:
                fout.write(words[w] + ' ')
                continue
            context_vec = np.zeros(len(glbmodel['time']))
            cnt = 0
            for i in range(max(0, w - args.window), min(len(words), w + args.window + 1)):
                if w == i:
                    continue
                if words[i] in glbmodel:
                    context_vec += glbmodel[words[i]]
                    cnt += 1
            if cnt == 0:
                fout.write(words[w] + ' ')
                continue
            context_vec = context_vec / cnt
            pb = prob(glbmodel, ctxmodel, words[w], context_vec)
            if len(pb) > 0:
                pos = np.argmax(pb)
            else:
                pos = 0
            choice = words[w] + '_s' + str(pos)
            if choice in true_sense:
                fout.write(true_sense[choice] + ' ')
            else:
                fout.write(words[w] + ' ')
        fout.write('\n')
        if idx % 10000 == 0:
            print idx, 'files processed!'
    fout.close()
