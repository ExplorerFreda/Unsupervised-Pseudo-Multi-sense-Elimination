#########################################################################
# File Name: sense_inductor.py
# Author: Haoyue Shi
# mail: freda.haoyue.shi@gmail.com
# Created Time: 2017-06-21 11:53
# Description: This script works as a sense inductor for SemEval 2007
#   WSI test. It's a part of out work in IJCNLP 2017.
#########################################################################

from eliminator import MultiSenseVectorSpace
import argparse
from gensim.models import KeyedVectors
from pprint import pprint
import json
import tester


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for pseudo multi-sense elimination by sense induction.')
    parser.add_argument('--global-model', type=str, default='',
                        help='global vector path for evaluation')
    parser.add_argument('--context-model', type=str, default='',
                        help='context vector path for evaluation')
    parser.add_argument('--local-model', type=str, default='',
                        help='local vector path for evaluation')
    parser.add_argument('--input-path', type=str, default='',
                        help='test path for the word sense induction task (refer to SemEval2007)')
    parser.add_argument('--output-path', type=str, default='',
                        help='output path for the word sense induction task (refer to SemEval2007)')
    parser.add_argument('--combine-pseudo', type=bool, default=True,
                        help='if True, output 20 different files w.r.t. the threshold of pseudo '
                             'multi-sense detection')

    args = parser.parse_args()
    if args.context_model == '':
        raise Exception('Must identify the context model.')
    if args.local_model == '':
        raise Exception('Must identify the local model.')
    if args.global_model == '':
        raise Exception('Must identify the global model.')
    if args.input_path == '':
        raise Exception('Must identify the input path.')
    if args.output_path == '':
        raise Exception('Must identify the output path.')

    glbmodel = KeyedVectors.load_word2vec_format(args.global_model)
    locmodel = KeyedVectors.load_word2vec_format(args.local_model)
    ctxmodel = KeyedVectors.load_word2vec_format(args.context_model)
    space = MultiSenseVectorSpace(glbmodel, locmodel, ctxmodel)

    fout = open(args.output_path, 'w')
    with open(args.input_path, 'r') as fin:
        lines = fin.readlines()
        fin.close()
        line_i = 0
        while line_i < len(lines):
            if lines[line_i].find('<instance') != -1:
                pos = lines[line_i].find('id=') + 4
                line = lines[line_i][pos:]
                pos = line.find('"')
                idx = line[:pos]
                line_i += 1
                sentence = lines[line_i]
                word, context = tester.sense_induction_test_word_with_context(idx, sentence)
                prob = tester.prob(glbmodel, ctxmodel, word, context)
                prototype = '.'.join(idx.split('.')[:-1])
                fout.write(prototype+' '+idx)
                for i in range(len(prob)):
                    fout.write(' '+prototype.split('.')[0]+('_s%d.%s/%.5f' % (i, prototype.split('.')[1], prob[i])))
                fout.write('\n')
            line_i += 1
    fout.close()
    if args.combine_pseudo:
        pseudo_multi_sense_pairs = space.detector.detect()
        true_cluster = dict()
        for item in locmodel.vocab:
            true_cluster[item] = item
        threshold = 1  # begin from combining nothing
        for step in range(0, 21):
            # update true_cluster
            for item in pseudo_multi_sense_pairs:
                if item[2] < threshold:
                    break
                u = item[0]
                v = item[1]
                if true_cluster[u] < true_cluster[v]:
                    cluster = true_cluster[u]
                else:
                    cluster = true_cluster[v]
                true_cluster[u] = cluster
                true_cluster[v] = cluster
            # output
            fout = open(args.output_path+('.%d' % step), 'w')
            for line in open(args.output_path):
                items = line.strip().split()
                fout.write(items[0] + ' ' + items[1])
                items = map(lambda x: (true_cluster[x.split('/')[0].split('.')[0]], float(x.split('/')[1])),
                            items[2:])
                weights = dict()
                for item in items:
                    if item[0] not in weights:
                        weights[item[0]] = item[1]
                    else:
                        weights[item[0]] += item[1]
                for item in weights:
                    fout.write(' %s/%.5f' % (item, weights[item]))
                fout.write('\n')

            # subtract threshold
            threshold -= 0.05
