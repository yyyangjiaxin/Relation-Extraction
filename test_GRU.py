from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
# from sklearn.utils.fixes import signature
from funcsigs import signature

FLAGS = tf.app.flags.FLAGS


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def main(_):
    # def main_for_evaluation():
    pathname = "./model/ATT_GRU_model-"

    wordembedding = np.load('./data/vec.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.num_classes = 4
    test_settings.big_num = 1
    # test_settings.big_num = 96    #################注意修改
    # test_settings.big_num=5561
    big_num_test = test_settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                # total_rel = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                    # for rel in rel_batch[i]:
                    #     total_rel.append(rel)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                # total_rel = np.array(total_rel)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch
                # feed_dict[mtest.input_rel] = total_rel

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)

            # testlist = range(1000, 1800, 100)
            # testlist = [9000]
            # testlist = [110]
            # testlist = range(85,171,5)
            testlist = [340]
            print('reading word embedding data...')
            vec = []
            word2id = {}
            f = open('./origin_data/vec.txt', encoding='utf-8')
            content = f.readline()
            content = content.strip().split()
            dim = int(content[1])
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)
            f.close()
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            print('reading relation to id')
            relation2id = {}
            id2relation = {}
            f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])
                id2relation[int(content[1])] = content[0]
            f.close()

            for model_iter in testlist:
                print('model_iter', model_iter)
                # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
                saver.restore(sess, pathname + str(model_iter))

                time_str = datetime.datetime.now().isoformat()
                print('Evaluating all test data and save data for PR curve')

                test_y = np.load('./data/testall_y.npy')
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')
                # test_rel = np.load('./data/testall_rel.npy')
                allprob = []  # 概率
                acc = []
                result = []  # 每个句子最后预测的关系类别
                # print(test_word)  #每句中每个字对应的id数组
                # print(len(test_word)) #结果为202  测试集中一共202条句子
                # print(np.array(test_word))
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))

                    for single_prob in prob:
                        top3_id = single_prob.argsort()[-3:][::-1]
                        a = top3_id[0]
                        result.append(a)
                        allprob.append(single_prob[1:])
                result_dict = {}  # 被正确判定的类别
                test_rel_dict = {}  # 测试集中每个类别的数量

                f1 = open('./data/test_q&a.txt', 'r', encoding='utf-8')
                lines = f1.readlines()
                i = 0
                for line in lines:
                    line = line.strip().split()
                    rel = int(line[3])
                    test_rel_dict[rel] = test_rel_dict.get(rel, 0) + 1
                    if rel == result[i]:
                        result_dict[rel] = result_dict.get(rel, 0) + 1
                    i = i + 1
                print('result_dict', result_dict)
                print('test_rel_dict', test_rel_dict)
                average_recall = 0
                for key, value in result_dict.items():
                    print(id2relation[key] + '的recall为：' + str(value / test_rel_dict[key]))
                    average_recall += value / test_rel_dict[key]
                print('平均recall为:' + str(average_recall / 4))

                allprob = np.reshape(np.array(allprob), (-1))
                order = np.argsort(-allprob)
                # print(allprob)

                print('saving all test result...')
                current_step = model_iter

                np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
                allans = np.load('./data/allans.npy')

                # caculate the pr curve area

                # print(len(allans)) #保存的是测试集中所有关系向量数组中的后三个数字所组成的一行数组  结果为202*3=606
                # print("－－－－:")
                # print(len(allprob))
                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))
                precision, recall, _ = precision_recall_curve(allans, allprob)
                # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
                step_kwargs = ({'step': 'post'}
                               if 'step' in signature(plt.fill_between).parameters
                               else {})
                plt.step(recall, precision, color='b', alpha=0.2,
                         where='post')
                plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.0])
                plt.xlim([0.2, 1.0])
                plt.title(str(model_iter) + 'Precision-Recall curve: AP={0:0.4f}'.format(
                    average_precision))
                plt.figure(1)  # plt.figure(1)是新建一个名叫 Figure1的画图窗口
                plt.plot(precision, recall)
                plt.savefig('p-r1.png')
                plt.show()

    # def main(_):

    # If you retrain the model, please remember to change the path to your own model below:
    # pathname = "./model/ATT_GRU_model-9000"
    pathname = "./model/ATT_GRU_model-340"

    wordembedding = np.load('./data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.num_classes = 4
    test_settings.big_num = 1

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                # total_rel = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                    # for rel in rel_batch[i]:
                    #     total_rel.append(rel)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                # total_rel = np.array(total_rel)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch
                # feed_dict[mtest.input_rel] = total_rel

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            print('reading word embedding data...')
            vec = []
            word2id = {}
            f = open('./origin_data/vec.txt', encoding='utf-8')
            content = f.readline()
            content = content.strip().split()
            dim = int(content[1])
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)
            f.close()
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            print('reading relation to id')
            relation2id = {}
            id2relation = {}
            f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])
                id2relation[int(content[1])] = content[0]
            f.close()

          
                # try:
                # BUG: Encoding error if user input directly from command line.
                line = input('请输入中文句子，格式为 "name1 name2 sentence":')
                # Read file from test file
                '''
                infile = open('test.txt', encoding='utf-8')
                line = ''
                for orgline in infile:
                    line = orgline.strip()
                    break
                infile.close()
                '''
                '''
                infile = open('./origin_data/test.txt', encoding='utf-8')
                line = ''
                for orgline in infile:
                    line = orgline.strip()
                    break
                infile.close()
                '''
                en1, en2, sentence = line.strip().split()
                print("实体1: " + en1)
                print("实体2: " + en2)
                print(sentence)
                relation = 0
                en1pos = sentence.find(en1)  # 找到实体在句子中的位置
                if en1pos == -1:
                    en1pos = 0
                en2pos = sentence.find(en2)
                if en2pos == -1:
                    en2post = 0
                output = []
                # length of sentence is 70
                fixlen = 70
                # max length of position embedding is 60 (-60~+60)
                maxlen = 60

                # Encoding test x  得到输入句子中每个字的id和距离两个实体的距离
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = pos_embed(i - en1pos)
                    rel_e2 = pos_embed(i - en2pos)
                    output.append([word, rel_e1, rel_e2])
                for i in range(min(fixlen, len(sentence))):

                    word = 0
                    if sentence[i] not in word2id:
                        # print(sentence[i])
                        # print('==')
                        word = word2id['UNK']
                        # print(word)
                    else:
                        # print(sentence[i])
                        # print('||')
                        word = word2id[sentence[i]]
                        # print(word)

                    output[i][0] = word
                test_x = []
                test_x.append([output])
                # print('test_x',test_x)

                # Encoding test y
                label = [0 for i in range(len(relation2id))]
                label[0] = 1
                test_y = []
                test_y.append(label)
                # print('test_y',test_y)

                test_x = np.array(test_x)
                # print('123')
                # print(test_x)
                test_y = np.array(test_y)
                # print(test_y)
                test_word = []
                test_pos1 = []
                test_pos2 = []

                for i in range(len(test_x)):
                    word = []
                    pos1 = []
                    pos2 = []
                    for j in test_x[i]:
                        temp_word = []
                        temp_pos1 = []
                        temp_pos2 = []
                        for k in j:
                            temp_word.append(k[0])
                            temp_pos1.append(k[1])
                            temp_pos2.append(k[2])
                        word.append(temp_word)
                        pos1.append(temp_pos1)
                        pos2.append(temp_pos2)
                    test_word.append(word)
                    test_pos1.append(pos1)
                    test_pos2.append(pos2)

                test_word = np.array(test_word)
                test_pos1 = np.array(test_pos1)
                test_pos2 = np.array(test_pos2)

                # print("test_word Matrix:")
                # print(test_word)
                # print("test_pos1 Matrix:")
                # print(test_pos1)
                # print("test_pos2 Matrix:")
                # print(test_pos2)

                prob, accuracy = test_step(test_word, test_pos1, test_pos2, test_y)
                prob = np.reshape(np.array(prob), (1, test_settings.num_classes))[0]
                print("关系是:")
                # print(prob)
                top3_id = prob.argsort()[-3:][::-1]
                for n, rel_id in enumerate(top3_id):
                    print("No." + str(n + 1) + ": " + id2relation[rel_id] + ", Probability is " + str(prob[rel_id]))
            # except Exception as e:
            #    print(e)

            # result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            # print(result)


import numpy

if __name__ == "__main__":
    # allans = []
    tf.app.run()
