# -*- coding: utf-8 -*-
# @Time    : 2019/10/31 19:37
# @Author  : Wang Qiaoling
# @File    : prepro.py
# @Software: PyCharm

import numpy as np
def get_word2id():
    word2id = {}
    print('reading word embedding data...')
    vec = []
    # Python中的花括号{}：代表dict字典数据类型，字典是Python中唯一内建的映射类型。字典中的值没有特殊的顺序，但都是存储在一个特定的键（key）下。键可以是数字、字符串甚至是元祖。
    f = open('./origin_data/vec.txt', encoding='utf-8')
    content = f.readline()
    content = content.strip().split()
    dim = int(content[1])
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    return word2id


def get_relation2id():
    relation2id = {}
    print('reading relation to id')
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()
    return relation2id


def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122

def readFile(path,relation2id,word2id, fixlen):
    train_sen = {}
    train_ans = {}

    print('reading data...')
    f = open(path, 'r', encoding='utf-8')

    while True:
        content = f.readline()

        if content == '':
            break

        content = content.strip().split()
        # get entity name
        en1 = content[0]  # 第一个实体
        en2 = content[1]  # 第二个实体
        # print("两个实体：", en1, en2)
        relation = 0

        tup = (en1, en2)

        label_tag = 0
        # if tup not in train_sen:
        train_sen[tup] = []
        train_sen[tup].append([])
        y_id = relation
        label_tag = 0
        label = [0 for i in range(len(relation2id))]
        label[y_id] = 1
        train_ans[tup] = []
        train_ans[tup].append(label)

        # else:
        #     y_id = relation
        #     label_tag = 0
        #     label = [0 for i in range(len(relation2id))]
        #     label[y_id] = 1
        #
        #     temp = find_index(label, train_ans[tup])
        #     if temp == -1:
        #         train_ans[tup].append(label)
        #         label_tag = len(train_ans[tup]) - 1
        #         train_sen[tup].append([])
        #     else:
        #         label_tag = temp

        # print("train_ans = ", train_ans)
        sentence = content[3]
        # print("sentence = ", sentence)

        en1pos = 0
        en2pos = 0

        # For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0

        # print("实体的位置：", en1pos, en2pos)
        output = []

        # Embeding the position
        for i in range(fixlen):
            # print("Embeding the position 循环")
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            # print([word, rel_e1, rel_e2])
            output.append([word, rel_e1, rel_e2])
        # print(len(output))

        # Embeding word2id
        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        # print(train_sen[tup][label_tag])
        # exit()

        train_sen[tup][label_tag].append(output)
    return train_ans, train_sen

def read_train_qafile(path, train_sen, train_ans):
    train_x = []
    train_y = []
    print('organizing data')
    f = open(path, 'w', encoding='utf-8')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    return train_x, train_y

fixlen = 2
word2id = get_word2id()
relation2id = get_relation2id()
train_ans, train_sen = readFile('./origin_data/train.txt', relation2id,word2id, fixlen)
test_ans, test_sen = readFile('./origin_data/test.txt', relation2id,word2id, fixlen)
train_x, train_y = read_train_qafile('./data/train_q&a.txt', train_sen,train_ans)
test_x, test_y = read_train_qafile('./data/train_q&a.txt', test_sen, test_ans)
print(len(train_sen))

# 将数据向量化
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# test_x = np.array(test_x)
# test_y = np.array(test_y)
# 将字典保存为npy文件
# np.save('./data/vec.npy', word2id)
# np.save('./data/train_x.npy', train_x)
# np.save('./data/train_y.npy', train_y)
# np.save('./data/testall_x.npy', test_x)
# np.save('./data/testall_y.npy', test_y)
