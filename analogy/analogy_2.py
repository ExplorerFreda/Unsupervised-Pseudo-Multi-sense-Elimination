#!/usr/bin/python  
# -*- coding:utf-8 -*-  

import gensim

topn = 1 # 通过gensim的most_similar函数检索最接近的topn个词

vocab_name = "./data/MSSG.50D.30K/MSSG.50D.30K.global" # 词典，实际上是global向量
log_name = "./data/MSSG.50D.30K/MSSG.50D.30K.log" # 对于每一个model，其回答错误的问题将会被输入到对应的log文件中
model_name = "./data/MSSG.50D.30K/MSSG.50D.30K.sense.py"

def exist(vocab, words) : # 判断一个问题中四个词有没有在词汇表中出现过
    for word in words :
        if word not in vocab : return 0
    return 1

def answer(model, w1, w2, w3, w4) : # 对一个问题进行检索
    for i in range(10) : # 枚举w1的每个词义
        if w1 + "_s" + str(i) not in model :
            break
        for j in range(10) : # 枚举w2的每个词义
            if w2 + "_s" + str(j) not in model :
                break
            for k in range(10) : # 枚举w3的每个词义
                if w3 + "_s" + str(k) not in model :
                    break
                ms = model.most_similar(positive = [w1 + "_s" + str(i), w3 + "_s" + str(k)], negative = [w2 + "_s" + str(j)], topn = topn) # 求与w1 - w2 + w3最接近的那个词向量，positive表示正的向量，negative表示负（需要减去）的向量，topn表示需要找最接近的topn个，等号前面的是函数的参数，等号后面的是在前面已经设定的，默认其值为1
                for item in ms : # 对于返回的词（理论上应该只有1个）
                    if item[0][:item[0].find("_s")] == w4 : # 如果这个词是w4
                        return 1
    return 0

def question(model, words) : # 对于给出的a b c d四个词，进行一次测试，只要四组测试有一组符合要求，我们就认为测试是成功的
    temp = answer(model, words[1], words[0], words[2], words[3]) # b - a + c = d
    temp |= answer(model, words[0], words[1], words[3], words[2]) # a - b + d = c
    temp |= answer(model, words[3], words[2], words[0], words[1]) # d - c + a = b
    temp |= answer(model, words[2], words[3], words[1], words[0]) # c - d + b = a
    return temp

def test(vocab, model, database) :
    total_number = 0
    total_correct = 0
    part_number = 0
    part_correct = 0
    log = open(log_name + "." + database, "w")

    for line in open("./test/" + database + ".txt" , "r") :
        if (line[0] == ":") : # 如果到了一个新的测试类
            if part_number != 0 : # 如果这不是第一个测试类
                print "    questions number : ", part_number
                print "        model correct number : ", part_correct
                print "        model accuracy : ", float(part_correct) / part_number
            print line[:-1]
            total_number += part_number # 数据统计与更新
            total_correct += part_correct
            part_number = 0
            part_correct = 0
            continue
        words = line[:-1].lower().split() # 获取一组问题
        if (exist(vocab, words) == 0) : continue # 如果有词语不存在于词典中
        if question(model, words) == 1 : part_correct += 1 # 如果回答正确就更新
        else : log.write(words[0] + " " + words[1] + " " + words[2] + " " + words[3] + "\n") # 如果回答错误就输出到相应log文件中
        part_number += 1
    print "    questions number : ", part_number
    print "        model correct number : ", part_correct
    print"         model accuracy : ", float(part_correct) / part_number
    total_number += part_number
    total_correct += part_correct
    print "%s questions number : " % database, total_number
    print "    model correct number : ", total_correct
    print "    model accuracy : ", float(total_correct) / total_number
    print ""

vocab = gensim.models.KeyedVectors.load_word2vec_format(vocab_name) # 读取词典
print "loading vocabulary successfully!"
model = gensim.models.KeyedVectors.load_word2vec_format(model_name)
print "loading model successfully!"
test(vocab, model, "semantic")
test(vocab, model, "syntactic")
